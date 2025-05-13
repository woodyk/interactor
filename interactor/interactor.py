#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: interactor.py
# Author: Wadih Khairallah
# Description: Universal AI interaction class
#              with streaming, tool calling,
#              dynamic model switching, async support,
#              and comprehensive error handling
# Created: 2025-03-14 12:22:57
# Modified: 2025-05-13 15:51:47

import os
import re
import sys
import json
import time
import uuid
import queue
import asyncio
import aiohttp
import inspect
import logging
import argparse
import tiktoken
import traceback
import threading
import subprocess

from typing import (
    Union,
    Dict,
    Tuple,
    Any,
    Optional,
    List,
    Callable,
    get_origin,
    get_args
)
from datetime import datetime, timezone

import openai
from openai import OpenAIError, RateLimitError, APIConnectionError

import anthropic
from anthropic import Anthropic, AsyncAnthropic

from rich.prompt import Confirm
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.syntax import Syntax
from rich.rule import Rule

from .session import Session

console = Console()
log = console.log
print = console.print


class Interactor:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "openai:gpt-4o-mini",
        fallback_model = "ollama:mistral-nemo:latest",
        tools: Optional[bool] = True,
        stream: bool = True,
        quiet: bool = False,
        context_length: int = 128000,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        log_path: Optional[str] = None,
        raw: Optional[bool] = False,
        session_enabled: bool = False,
        session_id: Optional[str] = None,
        session_path: Optional[str] = None
    ):
        """Initialize the universal AI interaction client.

        Args:
            base_url: Optional base URL for the API. If None, uses the provider's default URL.
            api_key: Optional API key. If None, attempts to use environment variables based on provider.
            model: Model identifier in format "provider:model_name" (e.g., "openai:gpt-4o-mini").
            tools: Enable (True) or disable (False) tool calling; None for auto-detection based on model support.
            stream: Enable (True) or disable (False) streaming responses.
            context_length: Maximum number of tokens to maintain in conversation history.
            max_retries: Maximum number of retries for failed API calls.
            retry_delay: Initial delay (in seconds) for exponential backoff retries.
            session_enabled: Enable persistent session support.
            session_id: Optional session ID to load messages from.

        Raises:
            ValueError: If provider is not supported or API key is missing for non-Ollama providers.
        """
        self.system = "You are a helpful Assistant."
        self.raw = raw
        self.quiet = quiet
        self.logger = logging.getLogger(f"InteractorLogger_{id(self)}")
        self.logger.setLevel(logging.DEBUG)
        self.providers = {
            "openai": {
                "sdk": "openai",
                "base_url": "https://api.openai.com/v1",
                "api_key": api_key or os.getenv("OPENAI_API_KEY") or None
            },
           "ollama": {
                "sdk": "openai",
                "base_url": "http://localhost:11434/v1",
                "api_key": api_key or "ollama"
            },
            "nvidia": {
                "sdk": "openai",
                "base_url": "https://integrate.api.nvidia.com/v1",
                "api_key": api_key or os.getenv("NVIDIA_API_KEY") or None
            },
            "google": {
                "sdk": "openai",
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
                "api_key": api_key or os.getenv("GEMINI_API_KEY") or None
            },
            "anthropic": {
                "sdk": "anthropic",
                "base_url": "https://api.anthropic.com/v1",
                "api_key": api_key or os.getenv("ANTHROPIC_API_KEY") or None
            },
            "mistral": {
                "sdk": "openai",
                "base_url": "https://api.mistral.ai/v1",
                "api_key": api_key or os.getenv("MISTRAL_API_KEY") or None
            },
            "deepseek": {
                "sdk": "openai",
                "base_url": "https://api.deepseek.com",
                "api_key": api_key or os.getenv("DEEPSEEK_API_KEY") or None
            },
            "grok": {
                "sdk": "grok",
                "base_url": "https://api.x.ai/v1",
                "api_key": api_key or os.getenv("GROK_API_KEY") or None
            }
        }


        # Console log handler (always enabled at WARNING+)
        if not self.logger.handlers:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.WARNING)
            console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
            self.logger.addHandler(console_handler)

        self._log_enabled = False
        if log_path:
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            ))
            self.logger.addHandler(file_handler)
            self._log_enabled = True


        self.token_estimate = 0
        self.last_token_estimate = 0
        self.stream = stream
        self.tools = []
        self.session_history = []
        self.history = []
        self.context_length = context_length
        self.encoding = None
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.reveal_tool = []
        self.fallback_model = fallback_model
        self.sdk = None

        # Session support
        self.session_enabled = session_enabled
        self.session_id = session_id
        self._last_session_id = session_id
        self.session = Session(directory=session_path) if session_enabled else None


        if model is None:
            model = "openai:gpt-4o-mini"

        # Initialize model + encoding
        self._setup_client(model, base_url, api_key)
        self.tools_enabled = self.tools_supported if tools is None else tools and self.tools_supported
        self._setup_encoding()
        self.messages_add(role="system", content=self.system)


    def _log(self, message: str, level: str = "info"):
        if self._log_enabled:
            getattr(self.logger, level)(message)


    def _setup_client(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """Initialize or reconfigure the Interactor for the given model and SDK.

        Ensures idempotent setup, assigns SDK-specific clients and tool handling logic,
        and normalizes history to match the provider-specific message schema.
        """
        if not model:
            raise ValueError("Model must be specified as 'provider:model_name'")

        provider, model_name = model.split(":", 1)

        if not hasattr(self, "session_history"):
            self.session_history = []

        # Skip setup if nothing has changed (client may not yet exist on first call)
        if (
            hasattr(self, "client")
            and self.client
            and self.provider == provider
            and self.model == model_name
            and self.base_url == (base_url or self.base_url)
        ):
            return

        if provider not in self.providers:
            raise ValueError(f"Unsupported provider: {provider}. Supported: {list(self.providers.keys())}")

        # Load provider configuration
        provider_config = self.providers[provider]
        self.sdk = provider_config.get("sdk", "openai")
        self.provider = provider
        self.model = model_name
        self.base_url = base_url or provider_config["base_url"]
        effective_api_key = api_key or provider_config["api_key"]

        if not effective_api_key and provider != "ollama":
            raise ValueError(f"API key not provided and not found in environment for {provider.upper()}_API_KEY")

        # SDK-specific configuration
        if self.sdk == "openai":
            self.client = openai.OpenAI(base_url=self.base_url, api_key=effective_api_key)
            self.async_client = openai.AsyncOpenAI(base_url=self.base_url, api_key=effective_api_key)
            self.sdk_runner = self._openai_runner
            self.tool_key = "tool_call_id"

        elif self.sdk == "anthropic":
            self.client = anthropic.Anthropic(api_key=effective_api_key)
            self.async_client = anthropic.AsyncAnthropic(api_key=effective_api_key)
            self.sdk_runner = self._anthropic_runner
            self.tool_key = "tool_use_id"

        else:
            raise ValueError(f"Unsupported SDK type: {self.sdk}")

        # Determine tool support
        self.tools_supported = self._check_tool_support()
        if not self.tools_supported:
            self.logger.warning(f"Tool calling not supported for {provider}:{model_name}")

        # Normalize session history to match SDK after any provider/model change
        self._normalizer(force=True)

        self._log(f"[MODEL] Switched to {provider}:{model_name}")


    def _check_tool_support(self) -> bool:
        """Determine if the current model supports tool calling.

        Returns:
            bool: True if tools are supported for the active provider/model, False otherwise.
        """
        try:
            if self.sdk == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": "Test tool support."}],
                    stream=False,
                    tools=[{
                        "type": "function",
                        "function": {
                            "name": "test_tool",
                            "description": "Check tool support",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"}
                                },
                                "required": ["query"]
                            }
                        }
                    }],
                    tool_choice="auto"
                )
                message = response.choices[0].message
                return bool(message.tool_calls and len(message.tool_calls) > 0)

            elif self.sdk == "anthropic":
                # For Claude models, we pre-define support based on model ID
                # Known tool-supporting Claude models
                claude_tool_models = ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku",
                                      "claude-3.5-sonnet", "claude-3.7-sonnet"]

                # Check if the current model supports tools
                for supported_model in claude_tool_models:
                    if supported_model in self.model.lower():
                        self._log(f"[TOOLS] Anthropic model {self.model} is known to support tools")
                        return True

                # If not explicitly supported, try to test
                try:
                    _ = self.client.messages.create(
                        model=self.model,
                        messages=[{"role": "user", "content": "What's the weather?"}],
                        tools=[{
                            "name": "test_tool",
                            "description": "Check tool support",
                            "input_schema": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"}
                                },
                                "required": ["query"]
                            }
                        }],
                        max_tokens=10
                    )
                    return True
                except anthropic.BadRequestError as e:
                    error_msg = str(e).lower()
                    if "tool" in error_msg and "not supported" in error_msg:
                        self._log(f"[TOOLS] Anthropic model {self.model} does not support tools: {e}")
                        return False
                    if "not a supported tool field" in error_msg:
                        self._log(f"[TOOLS] Anthropic API rejected tool format: {e}")
                        return False
                    raise
                except Exception as e:
                    self._log(f"[TOOLS] Unexpected error testing tool support: {e}", level="error")
                    return False

            else:
                self.logger.warning(f"Tool support check not implemented for SDK '{self.sdk}'")
                return False

        except Exception as e:
            self.logger.error(f"Tool support check failed for {self.provider}:{self.model} — {e}")
            return False


    def add_function(
        self,
        external_callable: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        override: bool = False,
        disabled: bool = False,
        schema_extensions: Optional[Dict[str, Any]] = None
    ):
        """
        Register a function for LLM tool calling with full type hints and metadata.

        Args:
            external_callable (Callable): The function to register.
            name (Optional[str]): Optional custom name. Defaults to function's __name__.
            description (Optional[str]): Optional custom description. Defaults to first line of docstring.
            override (bool): If True, replaces an existing tool with the same name.
            disabled (bool): If True, registers the function in a disabled state.
            schema_extensions (Optional[Dict[str, Any]]): Optional dictionary mapping parameter names to 
                schema extensions that override or add to the auto-generated schema.

        Raises:
            ValueError: If the callable is invalid or duplicate name found without override.

        Example:
            interactor.add_function(
                my_tool, 
                override=True, 
                disabled=False,
                schema_extensions={
                    "param1": {"minimum": 0, "maximum": 100},
                    "param2": {"format": "email"}
                }
            )
        """
        def _python_type_to_schema(ptype: Any) -> dict:
            """Convert a Python type annotation to OpenAI-compatible JSON Schema."""
            # Handle None case
            if ptype is None:
                return {"type": "null"}
            
            # Get the origin and arguments of the type
            origin = get_origin(ptype)
            args = get_args(ptype)
            
            # Handle Union types (including Optional)
            if origin is Union:
                # Check for Optional (Union with None)
                none_type = type(None)
                if none_type in args:
                    non_none = [a for a in args if a is not none_type]
                    if len(non_none) == 1:
                        inner = _python_type_to_schema(non_none[0])
                        inner_copy = inner.copy()
                        inner_copy["nullable"] = True
                        return inner_copy
                    # Multiple types excluding None
                    types = [_python_type_to_schema(a) for a in non_none]
                    return {"anyOf": types, "nullable": True}
                # Regular Union without None
                return {"anyOf": [_python_type_to_schema(a) for a in args]}
            
            # Handle List and similar container types
            if origin in (list, List):
                item_type = args[0] if args else Any
                if item_type is Any:
                    return {"type": "array"}
                return {"type": "array", "items": _python_type_to_schema(item_type)}
            
            # Handle Dict types with typing info
            if origin in (dict, Dict):
                if not args or len(args) != 2:
                    return {"type": "object"}
                
                key_type, val_type = args
                # We can only really use val_type in JSON Schema
                if val_type is not Any and val_type is not object:
                    return {
                        "type": "object",
                        "additionalProperties": _python_type_to_schema(val_type)
                    }
                return {"type": "object"}
            
            # Handle Literal types for enums
            if origin is Literal:
                values = args
                # Try to determine type from values
                if all(isinstance(v, str) for v in values):
                    return {"type": "string", "enum": list(values)}
                elif all(isinstance(v, bool) for v in values):
                    return {"type": "boolean", "enum": list(values)}
                elif all(isinstance(v, (int, float)) for v in values):
                    return {"type": "number", "enum": list(values)}
                else:
                    # Mixed types, use anyOf
                    return {"anyOf": [{"type": _get_json_type(v), "enum": [v]} for v in values]}
            
            # Handle basic types
            if ptype is str:
                return {"type": "string"}
            if ptype is int:
                return {"type": "integer"}
            if ptype is float:
                return {"type": "number"}
            if ptype is bool:
                return {"type": "boolean"}
            
            # Handle common datetime types
            if ptype is datetime:
                return {"type": "string", "format": "date-time"}
            if ptype is date:
                return {"type": "string", "format": "date"}
            
            # Handle UUID
            if ptype is uuid.UUID:
                return {"type": "string", "format": "uuid"}
            
            # Default to object for any other types
            return {"type": "object"}
        
        def _get_json_type(value):
            """Get the JSON Schema type name for a Python value."""
            if isinstance(value, str):
                return "string"
            elif isinstance(value, bool):
                return "boolean"
            elif isinstance(value, int) or isinstance(value, float):
                return "number"
            elif isinstance(value, list):
                return "array"
            elif isinstance(value, dict):
                return "object"
            else:
                return "object"  # Default
        
        def _parse_param_docs(docstring: str) -> dict:
            """Extract parameter descriptions from a docstring."""
            if not docstring:
                return {}
            
            lines = docstring.splitlines()
            param_docs = {}
            current_param = None
            in_params = False
            
            # Regular expressions for finding parameter sections and param lines
            param_section_re = re.compile(r"^(Args|Parameters):\s*$")
            param_line_re = re.compile(r"^\s{4}(\w+)\s*(?:\([^\)]*\))?:\s*(.*)")
            
            for line in lines:
                # Check if we're entering the parameters section
                if param_section_re.match(line.strip()):
                    in_params = True
                    continue
                    
                if in_params:
                    # Skip empty lines
                    if not line.strip():
                        continue
                        
                    # Check for a parameter definition line
                    match = param_line_re.match(line)
                    if match:
                        current_param = match.group(1)
                        param_docs[current_param] = match.group(2).strip()
                    # Check for continuation of a parameter description
                    elif current_param and line.startswith(" " * 8):
                        param_docs[current_param] += " " + line.strip()
                    # If we see a line that doesn't match our patterns, we're out of the params section
                    else:
                        current_param = None
            
            return param_docs
        
        # Start of main function logic
        
        # Skip if tools are disabled
        if not self.tools_enabled:
            return
            
        # Validate input callable
        if not external_callable:
            raise ValueError("A valid external callable must be provided.")
        
        # Set function name, either from parameter or from callable's __name__
        function_name = name or external_callable.__name__
        
        # Try to get docstring and extract description
        try:
            docstring = inspect.getdoc(external_callable)
            description = description or (docstring.split("\n")[0].strip() if docstring else "No description provided.")
        except Exception as e:
            self._log(f"[TOOL] Warning: Could not extract docstring from {function_name}: {e}", level="warning")
            docstring = ""
            description = description or "No description provided."
        
        # Extract parameter documentation from docstring
        param_docs = _parse_param_docs(docstring)
        
        # Handle conflicts with existing functions
        if override:
            self.delete_function(function_name)
        elif any(t["function"]["name"] == function_name for t in self.tools):
            raise ValueError(f"Function '{function_name}' is already registered. Use override=True to replace.")
        
        # Try to get function signature for parameter info
        try:
            signature = inspect.signature(external_callable)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot inspect callable '{function_name}': {e}")
        
        # Process parameters to build schema
        properties = {}
        required = []
        
        for param_name, param in signature.parameters.items():
            # Skip self, cls parameters for instance/class methods
            if param_name in ("self", "cls") and param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                continue
                
            # Get parameter annotation, defaulting to Any
            annotation = param.annotation if param.annotation != inspect.Parameter.empty else Any
            
            try:
                # Convert Python type to JSON Schema
                schema = _python_type_to_schema(annotation)
                
                # Add description from docstring or create a default one
                schema["description"] = param_docs.get(param_name, f"{param_name} parameter")
                
                # Add to properties
                properties[param_name] = schema
                
                # If no default value is provided, parameter is required
                if param.default == inspect.Parameter.empty:
                    required.append(param_name)
                    self._log(f"[TOOL] Parameter '{param_name}' is required", level="debug")
                else:
                    self._log(f"[TOOL] Parameter '{param_name}' has default value: {param.default}", level="debug")
                    
            except Exception as e:
                self._log(f"[TOOL] Error processing parameter {param_name} for {function_name}: {e}", level="error")
                # Add a basic object schema as fallback
                properties[param_name] = {
                    "type": "string",  # Default to string instead of object for better compatibility
                    "description": f"{param_name} parameter (type conversion failed)"
                }
                
                # For parameters with no default value, mark as required even if processing failed
                if param.default == inspect.Parameter.empty:
                    required.append(param_name)
                    self._log(f"[TOOL] Parameter '{param_name}' marked as required despite conversion failure", level="debug")
        
        # Apply schema extensions if provided
        if schema_extensions:
            for param_name, extensions in schema_extensions.items():
                if param_name in properties:
                    properties[param_name].update(extensions)
        
        # Create parameters object with proper placement of 'required' field
        parameters = {
            "type": "object",
            "properties": properties,
        }
        
        # Only add required field if there are required parameters
        if required:
            parameters["required"] = required
        
        # Build the final tool specification
        tool_spec = {
            "type": "function",
            "function": {
                "name": function_name,
                "description": description,
                "parameters": parameters
            }
        }
        
        # Set disabled flag if requested
        if disabled:
            tool_spec["function"]["disabled"] = True
        
        # Add to tools list
        self.tools.append(tool_spec)
        
        # Make the function available as an attribute on the instance
        setattr(self, function_name, external_callable)
        
        # Log the registration with detailed information
        self._log(f"[TOOL] Registered function '{function_name}' with {len(properties)} parameters", level="info")
        if required:
            self._log(f"[TOOL] Required parameters: {required}", level="info")
        
        return function_name  # Return the name for reference

        
    def disable_function(self, name: str) -> bool:
        """
        Disable a registered tool function by name.

        This marks the function as inactive for tool calling without removing it from the internal registry.
        The function remains visible in the tool listing but is skipped during tool selection by the LLM.

        Args:
            name (str): The name of the function to disable.

        Returns:
            bool: True if the function was found and disabled, False otherwise.

        Example:
            interactor.disable_function("extract_text")
        """
        for tool in self.tools:
            if tool["function"]["name"] == name:
                tool["function"]["disabled"] = True
                return True
        return False


    def enable_function(self, name: str) -> bool:
        """
        Re-enable a previously disabled tool function by name.

        This removes the 'disabled' flag from a tool function, making it available again for LLM use.

        Args:
            name (str): The name of the function to enable.

        Returns:
            bool: True if the function was found and enabled, False otherwise.

        Example:
            interactor.enable_function("extract_text")
        """
        for tool in self.tools:
            if tool["function"]["name"] == name:
                tool["function"].pop("disabled", None)
                return True
        return False


    def delete_function(self, name: str) -> bool:
        """
        Permanently remove a registered tool function from the Interactor.

        This deletes both the tool metadata and the callable attribute, making it fully inaccessible
        from the active session. Useful for dynamically trimming the toolset.

        Args:
            name (str): The name of the function to delete.

        Returns:
            bool: True if the function was found and removed, False otherwise.

        Example:
            interactor.delete_function("extract_text")
        """
        before = len(self.tools)
        self.tools = [tool for tool in self.tools if tool["function"]["name"] != name]
        if hasattr(self, name):
            delattr(self, name)
        return len(self.tools) < before


    def list_functions(self) -> List[Dict[str, Any]]:
        """Get the list of registered functions for tool calling.

        Returns:
            List[Dict[str, Any]]: List of registered functions.
        """
        return self.tools


    def list_models(
        self,
        providers: Optional[Union[str, List[str]]] = None,
        filter: Optional[str] = None
    ) -> List[str]:
        """Retrieve available models from configured providers.

        Args:
            providers: Provider name or list of provider names. If None, all are queried.
            filter: Optional regex to filter model names.

        Returns:
            List[str]: Sorted list of "provider:model_id" strings.
        """
        models = []

        if providers is None:
            providers_to_list = self.providers
        elif isinstance(providers, str):
            providers_to_list = {providers: self.providers.get(providers)}
        elif isinstance(providers, list):
            providers_to_list = {p: self.providers.get(p) for p in providers}
        else:
            return []

        invalid_providers = [p for p in providers_to_list if p not in self.providers or self.providers[p] is None]
        if invalid_providers:
            self.logger.error(f"Invalid providers: {invalid_providers}")
            return []

        regex_pattern = None
        if filter:
            try:
                regex_pattern = re.compile(filter, re.IGNORECASE)
            except re.error as e:
                self.logger.error(f"Invalid regex pattern: {e}")
                return []

        for provider_name, config in providers_to_list.items():
            sdk = config.get("sdk", "openai")
            base_url = config.get("base_url")
            api_key = config.get("api_key")

            try:
                if sdk == "openai":
                    client = openai.OpenAI(api_key=api_key, base_url=base_url)
                    response = client.models.list()
                    for model in response.data:
                        model_id = f"{provider_name}:{model.id}"
                        if not regex_pattern or regex_pattern.search(model_id):
                            models.append(model_id)

                elif sdk == "anthropic":
                    client = Anthropic(api_key=api_key)
                    response = client.models.list()
                    for model in response:
                        model_id = f"{provider_name}:{model.id}"
                        if not regex_pattern or regex_pattern.search(model_id):
                            models.append(model_id)
                else:
                    self.logger.warning(f"SDK '{sdk}' for provider '{provider_name}' is not supported by list_models()")

            except Exception as e:
                self.logger.error(f"Failed to list models for {provider_name}: {e}")

        return sorted(models, key=str.lower)


    async def _retry_with_backoff(self, func: Callable, *args, **kwargs):
        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)

            except (RateLimitError, APIConnectionError, aiohttp.ClientError) as e:
                if attempt == self.max_retries:
                    model_key = f"{self.provider}:{self.model}"
                    if self.fallback_model and model_key != self.fallback_model:
                        print(f"[yellow]Model '{model_key}' failed. Switching to fallback: {self.fallback_model}[/yellow]")
                        self._setup_client(self.fallback_model)
                        self._setup_encoding()
                        self._normalizer()
                        return await func(*args, **kwargs)  # retry once with fallback model
                    else:
                        self.logger.error(f"All {self.max_retries} retries failed: {e}")
                        raise

                delay = self.retry_delay * (2 ** attempt)
                self.logger.warning(f"Retry {attempt + 1}/{self.max_retries} after {delay}s due to {e}")
                self._log(f"[RETRY] Attempt {attempt + 1}/{self.max_retries} failed: {e}", level="warning")
                await asyncio.sleep(delay)

            except OpenAIError as e:
                self.logger.error(f"OpenAI error: {e}")
                raise

            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
                raise


    def interact(
        self,
        user_input: Optional[str],
        quiet: bool = False,
        tools: bool = True,
        stream: bool = True,
        markdown: bool = False,
        model: Optional[str] = None,
        output_callback: Optional[Callable[[str], None]] = None,
        session_id: Optional[str] = None,
        raw: Optional[bool] = None,
        tool_suppress: bool = True,
        timeout: float = 60.0
    ) -> Union[Optional[str], "TokenStream"]:
        """Main universal gateway for all LLM interaction.

        This function serves as the single entry point for all interactions with the language model.
        When `raw=False` (default), it handles the interaction internally and returns the full response.
        When `raw=True`, it returns a context manager that yields chunks of the response for custom handling.

        Args:
            user_input: Text input from the user.
            quiet: If True, don't print status info or progress.
            tools: Enable (True) or disable (False) tool calling.
            stream: Enable (True) or disable (False) streaming responses.
            markdown: If True, renders content as markdown.
            model: Optional model override.
            output_callback: Optional callback to handle the output.
            session_id: Optional session ID to load messages from.
            raw: If True, return a context manager instead of handling the interaction internally.
                 If None, use the class-level setting from __init__.
            tool_suppress: If True and raw=True, filter out tool-related status messages.
            timeout: Maximum time in seconds to wait for the stream to complete when raw=True.

        Returns:
            If raw=False: The complete response from the model as a string, or None if there was an error.
            If raw=True: A context manager that yields chunks of the response as they arrive.

        Example with default mode:
            response = ai.interact("Tell me a joke")

        Example with raw mode:
            with ai.interact("Tell me a joke", raw=True) as stream:
                for chunk in stream:
                    print(chunk, end="", flush=True)
        """
        if not user_input:
            return None

        if quiet or self.quiet:
            markdown = False
            stream = False

        # Determine if we should use raw mode
        # If raw parameter is explicitly provided, use that; otherwise use class setting
        use_raw = self.raw if raw is None else raw

        # If raw mode is requested, delegate to interact_raw
        if use_raw:
            return self.interact_raw(
                user_input=user_input,
                tools=tools,
                model=model,
                session_id=session_id,
                tool_suppress=tool_suppress,
                timeout=timeout
            )

        # Setup model if specified
        if model:
            self._setup_client(model)
            self._setup_encoding()

        # Session handling
        if self.session_enabled and session_id:
            self.session_id = session_id
            self.session_load(session_id)

        # Add user message using messages_add
        self.messages_add(role="user", content=user_input)

        # Log token count estimate
        token_count = self._count_tokens(self.history)
        if not quiet:
            print(f"[dim]Estimated tokens in context: {token_count} / {self.context_length}[/dim]")

        # Make sure we have enough context space
        if token_count > self.context_length:
            if self._cycle_messages():
                if not quiet:
                    print("[red]Context window exceeded. Cannot proceed.[/red]")
                return None

        # Log user input
        self._log(f"[USER] {user_input}")

        # Handle the actual interaction with complete streaming for all responses
        result = asyncio.run(self._interact_async_core(
            user_input=user_input,
            quiet=quiet,
            tools=tools,
            stream=stream,
            markdown=markdown,
            output_callback=output_callback
        ))

        # Log completion for this interaction
        self._log(f"[INTERACTION] Completed with {len(self.history)} total messages")

        return result


    def _interact_raw(
        self,
        user_input: Optional[str],
        tools: bool = True,
        model: Optional[str] = None,
        session_id: Optional[str] = None,
        tool_suppress: bool = True,
        timeout: float = 60.0
    ):
        """
        Low-level function that returns a raw stream of tokens from the model.
        
        This method works as a context manager that yields a generator of streaming tokens.
        The caller is responsible for handling the output stream. Typically, this is used
        indirectly through interact() with raw=True.
        
        Args:
            user_input: Text input from the user.
            tools: Enable (True) or disable (False) tool calling.
            model: Optional model override.
            session_id: Optional session ID to load messages from.
            tool_suppress: If True, filter out tool-related status messages.
            timeout: Maximum time in seconds to wait for the stream to complete.
            
        Returns:
            A context manager that yields a stream of tokens.
            
        Example:
            with ai.interact_raw("Hello world") as stream:
                for chunk in stream:
                    print(chunk, end="", flush=True)
        """
        if not user_input:
            return None

        # Setup model if specified
        if model:
            self._setup_client(model)
            self._setup_encoding()
        
        # Session handling
        if self.session_enabled and session_id:
            self.session_id = session_id
            self.session_load(session_id)
        
        # Add user message
        self.messages_add(role="user", content=user_input)
        
        # Log token count estimate
        token_count = self._count_tokens(self.history)
        self._log(f"[STREAM] Estimated tokens in context: {token_count} / {self.context_length}")
        
        # Make sure we have enough context space
        if token_count > self.context_length:
            if self._cycle_messages():
                self._log("[STREAM] Context window exceeded. Cannot proceed.", level="error")
                return None
        
        # Log user input
        self._log(f"[USER] {user_input}")
        
        # Create a token stream class using a thread-safe queue
        class TokenStream:
            def __init__(self, interactor, user_input, tools, tool_suppress, timeout):
                self.interactor = interactor
                self.user_input = user_input
                self.tools = tools
                self.tool_suppress = tool_suppress
                self.timeout = timeout
                self.token_queue = queue.Queue()
                self.thread = None
                self.result = None
                self.error = None
                self.completed = False
                
            def __enter__(self):
                # Start the thread for async interaction
                def stream_worker():
                    # Define output callback to put tokens in queue
                    def callback(text):
                        # Filter out tool messages if requested
                        if self.tool_suppress:
                            try:
                                # Check if this is a tool status message (JSON format)
                                data = json.loads(text)
                                if isinstance(data, dict) and data.get("type") == "tool_call":
                                    # Skip this message
                                    return
                            except (json.JSONDecodeError, TypeError):
                                # Not JSON or not a dict, continue normally
                                pass

                        # Add to queue
                        self.token_queue.put(text)

                    # Run the interaction in a new event loop
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    try:
                        # Run the interaction
                        self.result = loop.run_until_complete(
                            self.interactor._interact_async_core(
                                user_input=self.user_input,
                                quiet=True,
                                tools=self.tools,
                                stream=True,
                                markdown=False,
                                output_callback=callback
                            )
                        )
                        # Signal successful completion
                        self.completed = True
                    except Exception as e:
                        self.error = str(e)
                        self.interactor.logger.error(f"Streaming error: {traceback.format_exc()}")
                        # Add error information to the queue if we haven't yielded anything yet
                        if self.token_queue.empty():
                            self.token_queue.put(f"Error: {str(e)}")
                    finally:
                        # Signal end of stream regardless of success/failure
                        self.token_queue.put(None)
                        loop.close()

                # Start the worker thread
                self.thread = threading.Thread(target=stream_worker)
                self.thread.daemon = True
                self.thread.start()
                
                # Return self for iteration
                return self
            
            def __iter__(self):
                return self
            
            def __next__(self):
                # Get next token from queue with timeout to prevent hanging
                try:
                    token = self.token_queue.get(timeout=self.timeout)
                    if token is None:
                        # End of stream
                        raise StopIteration
                    return token
                except queue.Empty:
                    # Timeout reached
                    self.interactor.logger.warning(f"Stream timeout after {self.timeout}s")
                    if not self.completed and not self.error:
                        # Clean up the thread - it might be stuck
                        if self.thread and self.thread.is_alive():
                            # We can't forcibly terminate a thread in Python,
                            # but we can report the issue
                            self.interactor.logger.error("Stream worker thread is hung")
                    raise StopIteration
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                # Clean up resources
                if self.thread and self.thread.is_alive():
                    self.thread.join(timeout=2.0)
                    
                # Add messages to history if successful
                if self.completed and self.result and not exc_type:
                    if isinstance(self.result, str) and self.result != "No response.":
                        # If we had a successful completion, ensure the result is in the history
                        last_msg = self.interactor.history[-1] if self.interactor.history else None
                        if not last_msg or last_msg.get("role") != "assistant" or last_msg.get("content") != self.result:
                            # Add a clean assistant message to history if not already there
                            self.interactor.messages_add(role="assistant", content=self.result)
                            
                # If there was an error in the stream processing, log it
                if self.error:
                    self.interactor.logger.error(f"Stream processing error: {self.error}")
                    
                return False  # Don't suppress exceptions
        
        return TokenStream(self, user_input, tools, tool_suppress, timeout)


    async def _interact_async_core(
        self,
        user_input: str,
        quiet: bool = False,
        tools: bool = True,
        stream: bool = True,
        markdown: bool = False,
        output_callback: Optional[Callable] = None
    ) -> str:
        """Main SDK-agnostic async execution pipeline with tool call looping support."""
        # Prepare display handler
        live = Live(console=console, refresh_per_second=100) if markdown and stream else None
        if live:
            live.start()
        
        # Initialize variables for iteration tracking
        full_content = ""
        tool_enabled = self.tools_enabled and self.tools_supported and tools
        max_iterations = 5  # Prevent infinite loops
        iterations = 0
        
        # Main interaction loop - continues until no more tool calls or max iterations reached
        while iterations < max_iterations:
            iterations += 1
            
            try:
                # Execute the appropriate SDK runner - history is already normalized
                response_data = await self.sdk_runner(
                    model=self.model,
                    messages=self.history,
                    stream=stream,
                    markdown=markdown,
                    quiet=quiet if iterations == 1 else False,
                    live=live,
                    output_callback=output_callback
                )
                
                # Extract response data
                content = response_data.get("content", "")
                tool_calls = response_data.get("tool_calls", [])
                
                # Log the response data for debugging
                self._log(f"[ITERATION {iterations}] Content: {len(content)} chars, Tool calls: {len(tool_calls)}")
                
                # Add content to full response
                if iterations == 1:
                    full_content = content
                elif content:
                    if full_content and content:
                        full_content += f"\n{content}"
                    else:
                        full_content = content
                
                # Add assistant message with or without tool calls
                if tool_calls:
                    # Process each tool call
                    for call in tool_calls:
                        # Add assistant message with tool call
                        tool_info = {
                            "id": call["id"],
                            "name": call["function"]["name"],
                            "arguments": call["function"]["arguments"]
                        }
                        
                        # Add the assistant message with tool call
                        self.messages_add(
                            role="assistant", 
                            content=content if len(tool_calls) == 1 else "",
                            tool_info=tool_info
                        )
                        
                        # Execute the tool
                        call_name = call["function"]["name"]
                        call_args = call["function"]["arguments"]
                        call_id = call["id"]

                        # Stop Rich Live while executing tool calls 
                        live_was_active = True
                        if live and live.is_started:
                            live_was_active = True
                            live.stop()
                        
                        result = await self._handle_tool_call_async(
                            function_name=call_name,
                            function_arguments=call_args,
                            tool_call_id=call_id,
                            quiet=quiet,
                            safe=False,
                            output_callback=output_callback
                        )

                        # Restart live display if it was active before
                        if live_was_active and live:
                            live.start()
                        
                        # Add tool result message
                        tool_result_info = {
                            "id": call_id,
                            "result": result
                        }
                        
                        self.messages_add(
                            role="tool",
                            content=result,
                            tool_info=tool_result_info
                        )
                else:
                    # Simple assistant response without tool calls
                    self.messages_add(role="assistant", content=content)
                    break  # No more tools to process, we're done
                
                # Reset live display if needed
                if stream and live:
                    live.stop()
                    live = Live(console=console, refresh_per_second=100)
                    live.start()
            
            except Exception as e:
                self.logger.error(f"[{self.sdk.upper()} ERROR] {str(e)}")
                self._log(f"[ERROR] Error in interaction loop: {str(e)}", level="error")
                if live:
                    live.stop()
                return f"Error: {str(e)}"
        
        # Clean up display
        if live:
            live.stop()
        
        return full_content or None 


    async def _openai_runner(
        self,
        *,
        model,
        messages,
        stream,
        markdown=False,
        quiet=False,
        live=None,
        output_callback=None
    ):
        """Handle OpenAI-specific API interactions and response processing."""
        # Log what we're sending for debugging
        self._log(f"[OPENAI REQUEST] Sending request to {model} with {len(self.history)} messages", level="debug")

        # Prepare API parameters - history is already normalized by _normalizer
        params = {
            "model": model,
            "messages": self.history,
            "stream": stream,
        }

        # Add tools if enabled
        if self.tools_enabled and self.tools_supported:
            enabled_tools = self._get_enabled_tools()
            if enabled_tools:
                params["tools"] = enabled_tools
                params["tool_choice"] = "auto"

        # Call API with retry handling
        try:
            response = await self._retry_with_backoff(
                self.async_client.chat.completions.create,
                **params
            )
        except Exception as e:
            self.logger.error(f"[OPENAI ERROR RUNNER]: {traceback.format_exc()}")
            raise

        assistant_content = ""
        tool_calls_dict = {}

        # Process streaming response
        if stream and hasattr(response, "__aiter__"):
            async for chunk in response:
                delta = getattr(chunk.choices[0], "delta", None)
                finish_reason = getattr(chunk.choices[0], "finish_reason", None)

                # Handle content chunks
                if hasattr(delta, "content") and delta.content is not None:
                    text = delta.content
                    assistant_content += text
                    if output_callback:
                        output_callback(text)
                    elif live:
                        live.update(Markdown(assistant_content))
                    elif not markdown:
                        print(text, end="")

                # Process tool calls
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    for tool_call_delta in delta.tool_calls:
                        index = tool_call_delta.index
                        if index not in tool_calls_dict:
                            tool_calls_dict[index] = {
                                "id": tool_call_delta.id if hasattr(tool_call_delta, "id") else None,
                                "function": {"name": "", "arguments": ""}
                            }

                        function = getattr(tool_call_delta, "function", None)
                        if function:
                            name = getattr(function, "name", None)
                            args = getattr(function, "arguments", "")
                            if name:
                                tool_calls_dict[index]["function"]["name"] = name
                            if args:
                                tool_calls_dict[index]["function"]["arguments"] += args
                            if tool_call_delta.id and not tool_calls_dict[index]["id"]:
                                tool_calls_dict[index]["id"] = tool_call_delta.id

                        # Make sure the ID is set regardless
                        if hasattr(tool_call_delta, "id") and tool_call_delta.id and not tool_calls_dict[index]["id"]:
                            tool_calls_dict[index]["id"] = tool_call_delta.id

            if not output_callback and not markdown and not quiet:
                print()

        # Process non-streaming response
        else:
            message = response.choices[0].message
            assistant_content = message.content or ""

            if hasattr(message, "tool_calls") and message.tool_calls:
                for i, tool_call in enumerate(message.tool_calls):
                    tool_calls_dict[i] = {
                        "id": tool_call.id,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    }

            if output_callback:
                output_callback(assistant_content)
            elif not quiet:
                print(assistant_content)

        # Log tool calls for debugging
        if tool_calls_dict:
            self._log(f"[OPENAI TOOL CALLS] Found {len(tool_calls_dict)} tool calls", level="debug")
            for idx, call in tool_calls_dict.items():
                self._log(f"[OPENAI TOOL CALL {idx}] {call['function']['name']} with ID {call['id']}", level="debug")

        # Return standardized response format
        return {
            "content": assistant_content,
            "tool_calls": list(tool_calls_dict.values())
        }


    async def _anthropic_runner(
        self,
        *,
        model,
        messages,
        stream,
        markdown=False,
        quiet=False,
        live=None,
        output_callback=None
    ):
        """Handle Anthropic-specific API interactions and response processing."""
        # Log what we're sending for debugging
        self._log(f"[ANTHROPIC REQUEST] Sending request to {model} with {len(self.history)} messages", level="debug")
        
        # Prepare API parameters - history is already normalized by _normalizer
        params = {
            "model": model,
            "messages": self.history,
            "max_tokens": 8192,
            "system": self.system
        }
        
        # Add tools support if needed
        if self.tools_enabled and self.tools_supported:
            enabled_tools = []
            for tool in self._get_enabled_tools():
                # Extract parameters from OpenAI format
                tool_params = tool["function"]["parameters"]
                
                # Create Anthropic-compatible tool definition
                format_tool = {
                    "name": tool["function"]["name"],
                    "description": tool["function"].get("description", ""),
                    "input_schema": {
                        "type": "object",
                        "properties": tool_params.get("properties", {})
                    }
                }
                
                # Ensure 'required' is at the correct level for Anthropic (as a direct child of input_schema)
                if "required" in tool_params:
                    format_tool["input_schema"]["required"] = tool_params["required"]
                    
                enabled_tools.append(format_tool)
            
            params["tools"] = enabled_tools
        
        assistant_content = ""
        tool_calls_dict = {}
        
        try:
            # Process streaming response
            if stream:
                stream_params = params.copy()
                stream_params["stream"] = True
                
                stream_response = await self._retry_with_backoff(
                    self.async_client.messages.create,
                    **stream_params
                )
               
                content_type = None
                async for chunk in stream_response:
                    chunk_type = getattr(chunk, "type", "unknown")
                    self._log(f"[ANTHROPIC CHUNK] Type: {chunk_type}", level="debug")
                    if chunk_type == "content_block_start" and hasattr(chunk.content_block, "type"):
                        content_type = chunk.content_block.type
                        if content_type == "tool_use":
                            tool_id = chunk.content_block.id
                            tool_name = chunk.content_block.name
                            tool_input = chunk.content_block.input
                            tool_calls_dict[tool_id] = {
                                "id": tool_id,
                                "function": {
                                    "name": tool_name,
                                    "arguments": ""
                                }
                            }
                            self._log(f"[ANTHROPIC TOOL USE] {tool_name}", level="debug")
                    
                    # Handle text content
                    if chunk_type == "content_block_delta" and hasattr(chunk.delta, "text"):
                        delta = chunk.delta.text
                        assistant_content += delta
                        if output_callback:
                            output_callback(delta)
                        elif live:
                            live.update(Markdown(assistant_content))
                        elif not markdown:
                            print(delta, end="")

                    # Handle complete tool use
                    elif chunk_type == "content_block_delta" and content_type == "tool_use":
                        tool_calls_dict[tool_id]["function"]["arguments"] += chunk.delta.partial_json

            # Process non-streaming response
            else:
                # For non-streaming, ensure we don't send the stream parameter
                non_stream_params = params.copy()
                non_stream_params.pop("stream", None)  # Remove stream if it exists
                
                response = await self._retry_with_backoff(
                    self.async_client.messages.create,
                    **non_stream_params
                )

                # Extract text content
                for content_block in response.content:
                    if content_block.type == "text":
                        assistant_content += content_block.text

                    if content_block.type == "tool_use":
                        tool_id = content_block.id
                        tool_name = content_block.name
                        tool_input = content_block.input
                        tool_calls_dict[tool_id] = {
                            "id": tool_id,
                            "function": {
                                "name": tool_name,
                                "arguments": tool_input
                            }
                        }
                        self._log(f"[ANTHROPIC TOOL USE] {tool_name}", level="debug")
                
                if output_callback:
                    output_callback(assistant_content)
                elif not quiet:
                    print(assistant_content)
                    
        except Exception as e:
            self._log(f"[ANTHROPIC ERROR RUNNER] {traceback.format_exc()}", level="error")
            
            # Return something usable even in case of error
            return {
                "content": f"Error processing Anthropic response: {str(e)}",
                "tool_calls": []
            }
        
        # Return standardized response format
        return {
            "content": assistant_content,
            "tool_calls": list(tool_calls_dict.values())
        }


    def _get_enabled_tools(self) -> List[dict]:
        """Return the list of currently enabled tool function definitions."""
        return [
            tool for tool in self.tools
            if not tool["function"].get("disabled", False)
        ]


    async def _handle_tool_call_async(
        self,
        function_name: str,
        function_arguments: str,
        tool_call_id: str,
        quiet: bool = False,
        safe: bool = False,
        output_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """Process a tool call asynchronously and return the result.

        Args:
            function_name: Name of the function to call.
            function_arguments: JSON string containing the function arguments.
            tool_call_id: Unique identifier for this tool call.
            params: Parameters used for the original API call.
            safe: If True, prompts for confirmation before executing the tool call.
            output_callback: Optional callback to handle the tool call result.

        Returns:
            The result of the function call.

        Raises:
            ValueError: If the function is not found or JSON is invalid.
        """
        if isinstance(function_arguments, str):
            arguments = json.loads(function_arguments)
        else:
            arguments = function_arguments
        
        self._log(f"[TOOL:{function_name}] args={arguments}")

        func = getattr(self, function_name, None)
        if not func:
            raise ValueError(f"Function '{function_name}' not found.")

        be_quiet = self.quiet if quiet is None else quiet 

        if not be_quiet:
            print(f"\nRunning {function_name}...")

        if output_callback:
            notification = json.dumps({
                "type": "tool_call",
                "tool_name": function_name,
                "status": "started"
            })
            output_callback(notification)

        try:
            if safe:
                prompt = f"[bold yellow]Proposed tool call:[/bold yellow] {function_name}({json.dumps(arguments, indent=2)})\n[bold cyan]Execute? [y/n]: [/bold cyan]"
                confirmed = Confirm.ask(prompt, default=False)
                if not confirmed:
                    command_result = {
                        "status": "cancelled",
                        "message": "Tool call aborted by user"
                    }
                    print("[red]Tool call cancelled by user[/red]")
                else:
                    loop = asyncio.get_event_loop()
                    command_result = await loop.run_in_executor(None, lambda: func(**arguments))
            else:
                loop = asyncio.get_event_loop()
                command_result = await loop.run_in_executor(None, lambda: func(**arguments))

            try:
                json.dumps(command_result)
            except TypeError as e:
                self.logger.error(f"Tool call result not serializable: {e}")
                return {"error": "Tool call returned unserializable data."}

            if output_callback:
                notification = json.dumps({
                    "type": "tool_call",
                    "tool_name": function_name,
                    "status": "completed"
                })
                output_callback(notification)

            return command_result

        except Exception as e:
            self._log(f"[ERROR] Tool execution failed: {e}", level="error")
            self.logger.error(f"Error executing tool function '{function_name}': {e}")
            return {"error": str(e)}


    def _setup_encoding(self):
        """Set up the token encoding based on the current model."""
        try:
            if self.provider == "openai":
                try:
                    self.encoding = tiktoken.encoding_for_model(self.model)
                    self._log(f"[ENCODING] Loaded tokenizer for OpenAI model: {self.model}")
                except:
                    self.encoding = tiktoken.get_encoding("cl100k_base")
                    self._log(f"[ENCODING] Fallback to cl100k_base for model: {self.model}")
            else:
                self.encoding = tiktoken.get_encoding("cl100k_base")
                self._log(f"[ENCODING] Defaulting to cl100k_base for non-OpenAI model: {self.model}")
        except Exception as e:
            self.logger.error(f"Failed to setup encoding: {e}")
            self.encoding = tiktoken.get_encoding("cl100k_base")


    def _estimate_tokens_tiktoken(self, messages) -> int:
        """Rough token count estimate using tiktoken for OpenAI or fallback cases."""
        if not hasattr(self, "encoding") or not self.encoding:
            self._setup_encoding()
        return sum(len(self.encoding.encode(msg.get("content", ""))) for msg in messages if isinstance(msg.get("content"), str))


    def _count_tokens(self, messages, use_cache=True) -> int:
        """Accurately estimate token count for messages including tool calls with caching support.

        Args:
            messages: List of message objects in either OpenAI or Anthropic format.
            use_cache: Whether to use and update the token count cache.

        Returns:
            int: Estimated token count.
        """
        # Setup encoding if needed
        if not hasattr(self, "encoding") or not self.encoding:
            self._setup_encoding()
        
        # Initialize cache if it doesn't exist
        if not hasattr(self, "_token_count_cache"):
            self._token_count_cache = {}
        
        # Generate a cache key based on message content hashes
        if use_cache:
            try:
                # Create a cache key using message IDs if available, or content hashes
                cache_key_parts = []
                for msg in messages:
                    if isinstance(msg, dict):
                        # Try to use stable identifiers for cache key
                        msg_id = msg.get("id", None)
                        timestamp = msg.get("timestamp", None)
                        
                        if msg_id and timestamp:
                            cache_key_parts.append(f"{msg_id}:{timestamp}")
                        else:
                            # Fall back to content-based hash if no stable IDs
                            content_str = str(msg.get("content", ""))
                            role = msg.get("role", "unknown")
                            cache_key_parts.append(f"{role}:{hash(content_str)}")
                
                cache_key = ":".join(cache_key_parts)
                if cache_key in self._token_count_cache:
                    return self._token_count_cache[cache_key]
            except Exception as e:
                # If caching fails, just continue with normal counting
                self._log(f"[TOKEN COUNT] Cache key generation failed: {e}", level="debug")
                use_cache = False
        
        # For Claude models, try to use their built-in token counter
        if self.sdk == "anthropic":
            try:
                # Convert messages to Anthropic format if needed
                anthropic_messages = []
                for msg in messages:
                    if msg.get("role") == "system":
                        continue  # System handled separately
                    
                    if msg.get("role") == "tool":
                        # Skip tool messages in token count to avoid double-counting
                        continue
                    
                    if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                        # Already in Anthropic format with tool_result
                        anthropic_messages.append(msg)
                    elif msg.get("role") in ["user", "assistant"]:
                        if not msg.get("tool_calls") and not msg.get("tool_use"):
                            # Simple message
                            anthropic_messages.append({
                                "role": msg.get("role"),
                                "content": msg.get("content", "")
                            })
                
                # Use Anthropic's token counter if messages exist
                if anthropic_messages:
                    response = self.client.messages.count_tokens(
                        model=self.model,
                        messages=anthropic_messages,
                        system=self.system
                    )
                    token_count = response.input_tokens
                    
                    # Cache the result for future use
                    if use_cache and 'cache_key' in locals():
                        self._token_count_cache[cache_key] = token_count
                    
                    return token_count
            except Exception as e:
                # Fall back to our estimation
                self._log(f"[TOKEN COUNT] Error using Anthropic token counter: {e}", level="debug")
        
        # More accurate token counting for all message types
        num_tokens = 0
        
        # Count tokens for each message
        for msg in messages:
            # Base token count for message metadata (role + message format)
            num_tokens += 4  # Message overhead
            
            # Add tokens for role name
            role = msg.get("role", "")
            num_tokens += len(self.encoding.encode(role))
            
            # Count tokens in message content
            if isinstance(msg.get("content"), str):
                content = msg.get("content", "")
                content_tokens = len(self.encoding.encode(content))
                num_tokens += content_tokens
                
            elif isinstance(msg.get("content"), list):
                # Handle Anthropic-style content lists
                for item in msg.get("content", []):
                    if isinstance(item, dict):
                        # Tool result or other structured content
                        if item.get("type") == "tool_result":
                            result_content = item.get("content", "")
                            if isinstance(result_content, str):
                                num_tokens += len(self.encoding.encode(result_content))
                            else:
                                # JSON serialization for dict/list content
                                num_tokens += len(self.encoding.encode(json.dumps(result_content)))
                            # Add tokens for tool_use_id and type fields
                            num_tokens += len(self.encoding.encode(item.get("type", "")))
                            num_tokens += len(self.encoding.encode(item.get("tool_use_id", "")))
                        
                        # Text content type
                        elif item.get("type") == "text":
                            num_tokens += len(self.encoding.encode(item.get("text", "")))
                        
                        # Tool use type
                        elif item.get("type") == "tool_use":
                            num_tokens += len(self.encoding.encode(item.get("name", "")))
                            tool_input = item.get("input", {})
                            if isinstance(tool_input, str):
                                num_tokens += len(self.encoding.encode(tool_input))
                            else:
                                num_tokens += len(self.encoding.encode(json.dumps(tool_input)))
                            num_tokens += len(self.encoding.encode(item.get("id", "")))
                    else:
                        # Plain text content
                        num_tokens += len(self.encoding.encode(str(item)))
            
            # Count tokens in tool calls for OpenAI format
            if msg.get("tool_calls"):
                for tool_call in msg.get("tool_calls", []):
                    if isinstance(tool_call, dict):
                        # Count tokens for function name
                        func_name = tool_call.get("function", {}).get("name", "")
                        num_tokens += len(self.encoding.encode(func_name))
                        
                        # Count tokens for arguments
                        args = tool_call.get("function", {}).get("arguments", "")
                        if isinstance(args, str):
                            num_tokens += len(self.encoding.encode(args))
                        else:
                            num_tokens += len(self.encoding.encode(json.dumps(args)))
                        
                        # Add tokens for id and type fields
                        num_tokens += len(self.encoding.encode(tool_call.get("id", "")))
                        num_tokens += len(self.encoding.encode(tool_call.get("type", "function")))
            
            # Count tokens in Anthropic tool_use field
            if msg.get("tool_use"):
                tool_use = msg.get("tool_use")
                # Count tokens for name
                num_tokens += len(self.encoding.encode(tool_use.get("name", "")))
                
                # Count tokens for input
                tool_input = tool_use.get("input", {})
                if isinstance(tool_input, str):
                    num_tokens += len(self.encoding.encode(tool_input))
                else:
                    num_tokens += len(self.encoding.encode(json.dumps(tool_input)))
                
                # Add tokens for id field
                num_tokens += len(self.encoding.encode(tool_use.get("id", "")))
            
            # Handle tool response message format
            if msg.get("role") == "tool":
                # Add tokens for tool_call_id
                tool_id = msg.get("tool_call_id", "")
                num_tokens += len(self.encoding.encode(tool_id))
        
        # Add message end tokens
        num_tokens += 2
        
        # Cache the result for future use
        if use_cache and 'cache_key' in locals():
            self._token_count_cache[cache_key] = num_tokens
        
        return num_tokens


    def _cycle_messages(self):
        """
        Intelligently trim the message history to fit within the allowed context length.

        This method implements a sophisticated trimming strategy:
        1. Always preserves system messages
        2. Always keeps the most recent complete conversation turn
        3. Prioritizes keeping tool call chains intact
        4. Preserves important context from earlier exchanges
        5. Aggressively prunes redundant information before essential content

        Returns:
            bool: True if all messages were trimmed (context exceeded), False otherwise.
        """
        # Check if we need to trim
        token_count = self._count_tokens(self.history)

        # If we're already under the limit, return early
        if token_count <= self.context_length:
            return False

        self._log(f"[TRIM] Starting message cycling: {token_count} tokens exceeds {self.context_length} limit", level="info")

        # We'll need to track tokens as we reconstruct the history
        remaining_tokens = token_count
        target_tokens = max(self.context_length * 0.8, self.context_length - 1000)  # Target 80% or 1000 less than max

        # First pass: identify critical messages we must keep
        must_keep = []
        tool_chain_groups = {}  # Group related tool calls and their results

        # Always keep system messages (should be first)
        system_indices = []
        for i, msg in enumerate(self.history):
            if msg.get("role") == "system":
                system_indices.append(i)
                must_keep.append(i)

        # Identify the most recent complete exchange (user question + assistant response)
        latest_exchange = []
        # Start from the end and work backward to find the last complete exchange
        for i in range(len(self.history) - 1, -1, -1):
            msg = self.history[i]
            if msg.get("role") == "assistant" and not latest_exchange:
                latest_exchange.append(i)
            elif msg.get("role") == "user" and latest_exchange:
                latest_exchange.append(i)
                break

        # Add the latest exchange to must-keep
        must_keep.extend(latest_exchange)

        # Identify tool chains - track which messages belong to the same tool flow
        tool_id_to_chain = {}
        for i, msg in enumerate(self.history):
            # For assistant messages with tool calls
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tool_call in msg.get("tool_calls"):
                    tool_id = tool_call.get("id")
                    if tool_id:
                        tool_id_to_chain[tool_id] = tool_id_to_chain.get(tool_id, []) + [i]

            # For tool response messages
            elif msg.get("role") == "tool" and msg.get("tool_call_id"):
                tool_id = msg.get("tool_call_id")
                tool_id_to_chain[tool_id] = tool_id_to_chain.get(tool_id, []) + [i]

            # For Anthropic format with tool use
            elif msg.get("role") == "assistant" and isinstance(msg.get("content"), list):
                for block in msg.get("content", []):
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        tool_id = block.get("id")
                        if tool_id:
                            tool_id_to_chain[tool_id] = tool_id_to_chain.get(tool_id, []) + [i]

            # For Anthropic tool result messages
            elif msg.get("role") == "user" and isinstance(msg.get("content"), list):
                for block in msg.get("content", []):
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        tool_id = block.get("tool_use_id")
                        if tool_id:
                            tool_id_to_chain[tool_id] = tool_id_to_chain.get(tool_id, []) + [i]

        # Group together all indices for each tool chain
        for tool_id, indices in tool_id_to_chain.items():
            chain_key = f"tool_{min(indices)}"  # Group by the earliest message
            if chain_key not in tool_chain_groups:
                tool_chain_groups[chain_key] = set()
            tool_chain_groups[chain_key].update(indices)

        # Second pass: calculate tokens for each message
        message_tokens = []
        for i, msg in enumerate(self.history):
            # Count tokens for this individual message
            tokens = self._count_tokens([msg])
            message_tokens.append((i, tokens))

        # Keep the messages identified as must-keep
        keep_indices = set(must_keep)

        # Calculate the tokens we've committed to keeping
        keep_tokens = sum(tokens for i, tokens in message_tokens if i in keep_indices)

        # Check if we've already exceeded the target with just must-keep messages
        if keep_tokens > self.context_length:
            # We're in trouble - the essential messages alone exceed context
            # Drop older messages until we're under the limit
            all_indices = sorted(keep_indices)

            # Start dropping oldest messages, but NEVER drop system messages
            for idx in all_indices:
                if idx not in system_indices:
                    keep_indices.remove(idx)
                    keep_tokens -= message_tokens[idx][1]
                    if keep_tokens <= target_tokens:
                        break

            # If we've removed everything but system messages and still over limit
            if keep_tokens > self.context_length:
                self._log(f"[TRIM] Critical failure: even with minimal context ({keep_tokens} tokens), we exceed the limit", level="error")
                # Keep only system messages if any
                keep_indices = set(system_indices)
                return True  # Context exceeded completely

        # Third pass: keep the most important tool chains intact
        available_tokens = target_tokens - keep_tokens
        # Sort tool chains by recency (assumed by the chain_key which uses the earliest message)
        sorted_chains = sorted(tool_chain_groups.items(), key=lambda x: x[0], reverse=True)

        for chain_key, indices in sorted_chains:
            # Skip if we've already decided to keep all messages in this chain
            if indices.issubset(keep_indices):
                continue

            # Calculate how many tokens this chain would add
            chain_tokens = sum(tokens for i, tokens in message_tokens if i in indices and i not in keep_indices)

            # If we can fit the entire chain, keep it
            if chain_tokens <= available_tokens:
                keep_indices.update(indices)
                available_tokens -= chain_tokens
            # Otherwise, we might want to keep partial chains in the future, but for now, skip

        # Fourth pass: fill in with as many remaining messages as possible, prioritizing recency
        # Get remaining messages sorted by recency (newest first)
        remaining_indices = [(i, tokens) for i, tokens in message_tokens if i not in keep_indices]
        remaining_indices.sort(reverse=True)  # Sort newest first

        for i, tokens in remaining_indices:
            if tokens <= available_tokens:
                keep_indices.add(i)
                available_tokens -= tokens

        # Final message reconstruction
        self._log(f"[TRIM] Keeping {len(keep_indices)}/{len(self.history)} messages, estimated {target_tokens - available_tokens} tokens", level="info")

        # Create new history with just the kept messages, preserving order
        new_history = [self.history[i] for i in sorted(keep_indices)]
        self.history = new_history

        # Update session_history to match the pruned history
        if hasattr(self, "session_history"):
            # Map between history items and session_history
            session_to_keep = []

            # For each session history message, check if it corresponds to a kept message
            for session_msg in self.session_history:
                # Keep system messages
                if session_msg.get("role") == "system":
                    session_to_keep.append(session_msg)
                    continue

                # Try to match based on available IDs or content
                msg_id = session_msg.get("id")

                # For tool messages, check tool_info.id against tool_call_id
                if "metadata" in session_msg and "tool_info" in session_msg["metadata"]:
                    tool_id = session_msg["metadata"]["tool_info"].get("id")

                    # Check if this tool_id is still in the kept history
                    for history_msg in new_history:
                        # Check standard ids
                        history_tool_id = None

                        # Check OpenAI format
                        if history_msg.get("role") == "tool":
                            history_tool_id = history_msg.get("tool_call_id")
                        elif history_msg.get("role") == "assistant" and history_msg.get("tool_calls"):
                            for call in history_msg.get("tool_calls", []):
                                if call.get("id") == tool_id:
                                    history_tool_id = call.get("id")
                                    break

                        # Check Anthropic format
                        elif isinstance(history_msg.get("content"), list):
                            for block in history_msg.get("content", []):
                                if isinstance(block, dict):
                                    if block.get("type") == "tool_use" and block.get("id") == tool_id:
                                        history_tool_id = block.get("id")
                                        break
                                    elif block.get("type") == "tool_result" and block.get("tool_use_id") == tool_id:
                                        history_tool_id = block.get("tool_use_id")
                                        break

                        if history_tool_id == tool_id:
                            session_to_keep.append(session_msg)
                            break

                # For regular messages, try content matching as fallback
                else:
                    content_match = False
                    if isinstance(session_msg.get("content"), str) and session_msg.get("content"):
                        for history_msg in new_history:
                            if history_msg.get("role") == session_msg.get("role") and history_msg.get("content") == session_msg.get("content"):
                                content_match = True
                                break

                    if content_match:
                        session_to_keep.append(session_msg)

            # Update session_history with kept messages
            self.session_history = session_to_keep

            # Re-normalize to ensure consistency
            self._normalizer(force=True)

        # Verify our final token count
        final_token_count = self._count_tokens(self.history)
        self._log(f"[TRIM] Final history has {len(self.history)} messages, {final_token_count} tokens", level="info")

        # Return whether we've completely exceeded context
        return final_token_count > self.context_length or len(self.history) == len(system_indices)


    def messages_add(
        self,
        role: str,
        content: Any,
        tool_info: Optional[Dict] = None,
        normalize: bool = True
    ) -> str:
        """
        Add a message to the standardized session_history and then update SDK-specific history.
        
        This method is the central point for all message additions to the conversation.
        
        Args:
            role: The role of the message ("user", "assistant", "system", "tool")
            content: The message content (text or structured)
            tool_info: Optional tool-related metadata
            normalize: Whether to normalize history after adding this message
            
        Returns:
            str: Unique ID of the added message
        """
        # Generate a unique message ID
        message_id = str(uuid.uuid4())
        
        # Create the standardized message for session_history
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Store system messages directly
        if role == "system":
            self.system = content
        
        # Create standard format message
        standard_message = {
            "role": role,
            "content": content,
            "id": message_id,
            "timestamp": timestamp,
            "metadata": {
                "sdk": self.sdk
            }
        }
        
        # Add tool info if provided
        if tool_info:
            standard_message["metadata"]["tool_info"] = tool_info
        
        # Add to session_history
        if not hasattr(self, "session_history"):
            self.session_history = []
        
        self.session_history.append(standard_message)
        
        # Save to persistent session if enabled
        if self.session_enabled and self.session_id:
            # Convert standard message to session-compatible format
            session_msg = {
                "role": role,
                "content": content,
                "id": message_id,
                "timestamp": timestamp
            }
            
            # Add tool-related fields if present
            if tool_info:
                for key, value in tool_info.items():
                    session_msg[key] = value
            
            # Store in session
            self.session.msg_insert(self.session_id, session_msg)
        
        # Update the SDK-specific format in self.history by running the normalizer
        if normalize:
            # We only need to normalize the most recent message for efficiency
            # Pass a flag indicating we're just normalizing a new message
            self._normalizer(force=False, new_message_only=True)
        
        # Log the added message
        self._log(f"[MESSAGE ADDED] {role}: {str(content)[:50]}...")
        
        return message_id


    def messages_system(self, prompt: str):
        """Set or retrieve the current system prompt."""
        if not isinstance(prompt, str) or not prompt:
            return self.system

        # If the prompt hasn't changed, don't do anything
        if self.system == prompt:
            return self.system

        # Update the system prompt
        old_system = self.system
        self.system = prompt

        # For OpenAI, update or insert the system message in history
        if self.sdk == "openai":
            # Check if there's already a system message
            system_index = next((i for i, msg in enumerate(self.history)
                                if msg.get("role") == "system"), None)

            if system_index is not None:
                # Update existing system message
                self.history[system_index]["content"] = prompt
            else:
                # Insert new system message at the beginning
                self.history.insert(0, {"role": "system", "content": prompt})

        # For Anthropic, system message is not part of history, just save it for API calls

        # Log to session only if prompt actually changed
        if self.session_enabled and self.session_id and old_system != prompt:
            self.session.msg_insert(self.session_id, {"role": "system", "content": prompt})

        return self.system


    def messages(self) -> list:
        """Return full session messages (persisted or in-memory)."""
        if self.session_enabled and self.session_id:
            return self.session.load_full(self.session_id).get("messages", [])
        return self.session_history


    def messages_length(self) -> int:
        """Calculate the total token count for the message history."""
        if not self.encoding:
            return 0

        total_tokens = 0
        for message in self.history:
            if message.get("content"):
                total_tokens += len(self.encoding.encode(message["content"]))
            if message.get("tool_calls"):
                for tool_call in message["tool_calls"]:
                    if tool_call.get("function"):
                        total_tokens += len(self.encoding.encode(tool_call["function"].get("name", "")))
                        total_tokens += len(self.encoding.encode(tool_call["function"].get("arguments", "")))
        return total_tokens


    def session_load(self, session_id: Optional[str]):
        """Load and normalize messages for a specific session."""
        self.session_id = session_id
        self._last_session_id = session_id
        
        if self.session_enabled and session_id:
            try:
                # Load raw session data
                session_data = self.session.load_full(session_id)
                messages = session_data.get("messages", [])
                
                # Convert session format to our standard format
                self.session_history = []
                
                # Track the most recent system message
                latest_system_msg = None
                
                for msg in messages:
                    # Extract fields
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    msg_id = msg.get("id", str(uuid.uuid4()))
                    timestamp = msg.get("timestamp", datetime.now(timezone.utc).isoformat())
                    
                    # If this is a system message, track it but don't add to session_history yet
                    if role == "system":
                        if latest_system_msg is None or timestamp > latest_system_msg["timestamp"]:
                            latest_system_msg = {
                                "role": role,
                                "content": content,
                                "id": msg_id,
                                "timestamp": timestamp,
                                "metadata": {"sdk": self.sdk}
                            }
                        continue
                    
                    # Build tool_info if present
                    tool_info = None
                    if any(key in msg for key in ["tool_use_id", "tool_call_id", "name", "arguments"]):
                        tool_info = {
                            "id": msg.get("tool_use_id") or msg.get("tool_call_id"),
                            "name": msg.get("name", "unknown_tool"),
                            "arguments": msg.get("arguments", {})
                        }
                    
                    # Create standard message
                    standard_msg = {
                        "role": role,
                        "content": content,
                        "id": msg_id,
                        "timestamp": timestamp,
                        "metadata": {
                            "sdk": self.sdk
                        }
                    }
                    
                    if tool_info:
                        standard_msg["metadata"]["tool_info"] = tool_info
                    
                    self.session_history.append(standard_msg)
                
                # If we found a system message, update the system property and add to history
                if latest_system_msg:
                    self.system = latest_system_msg["content"]
                    # Insert at the beginning of session_history
                    self.session_history.insert(0, latest_system_msg)
                else:
                    # If no system message was found, add the current system message
                    self.messages_add(role="system", content=self.system)
                
                # Normalize to current SDK format
                self._normalizer(force=True)
                
                self._log(f"[SESSION] Switched to session '{session_id}'")
            except Exception as e:
                self.logger.error(f"Failed to load session '{session_id}': {e}")
                self.session_reset()
        else:
            # Reset to empty state with system message
            self.session_reset()


    def session_reset(self):
        """
        Reset the current session state and reinitialize to default system prompt.

        Clears history, disables session ID tracking, and returns to in-memory mode.
        """
        self.session_id = None
        self._last_session_id = None
        
        # Clear histories
        self.session_history = []
        self.history = []
        
        # Reapply the system message
        if hasattr(self, "system") and self.system:
            # Add to session_history
            self.messages_add(role="system", content=self.system)
        else:
            # Ensure we have a default system message
            self.system = "You are a helpful Assistant."
            self.messages_add(role="system", content=self.system)
        
        self._log("[SESSION] Reset to in-memory mode")


    def _normalizer(self, force=False, new_message_only=False):
        """
        Central normalization function that transforms the standard session_history
        into the SDK-specific format needed in self.history.
        
        Args:
            force (bool): If True, always normalize even if SDK hasn't changed.
                         Default is False, which only normalizes on SDK change.
            new_message_only (bool): If True, only normalize the most recent message
                                   for efficiency when adding single messages.
        """
        # Skip normalization if SDK hasn't changed and force is False
        if not force and hasattr(self, '_last_sdk') and self._last_sdk == self.sdk:
            # If we only need to normalize the most recent message
            if new_message_only and self.session_history:
                # Get the most recent message from session_history
                recent_msg = self.session_history[-1]
                
                # Apply SDK-specific normalization for just this message
                if self.sdk == "openai":
                    self._openai_normalize_message(recent_msg)
                elif self.sdk == "anthropic":
                    self._anthropic_normalize_message(recent_msg)
                else:
                    # Generic handler for unknown SDKs
                    self._generic_normalize_message(recent_msg)
                    
                return
        
        # Record the current SDK to detect future changes
        self._last_sdk = self.sdk
        
        # For full normalization, clear current history and rebuild it
        self.history = []
        
        # Call the appropriate SDK-specific normalizer
        if self.sdk == "openai":
            self._openai_normalizer()
        elif self.sdk == "anthropic":
            self._anthropic_normalizer()
        else:
            self.logger.warning(f"No normalizer available for SDK: {self.sdk}")
            # Fallback to a simple conversion for unknown SDKs
            for msg in self.session_history:
                self._generic_normalize_message(msg)


    def _openai_normalizer(self):
        """
        Convert standardized session_history to OpenAI-compatible format in self.history.
        """
        # For OpenAI, we need to include system message in the history
        # and convert tool calls/results to OpenAI format

        # Start with empty history
        self.history = []

        # First, add the current system message at position 0
        self.history.append({
            "role": "system",
            "content": self.system
        })

        # Process all non-system messages
        for msg in self.session_history:
            if msg["role"] == "system":
                continue  # Skip system messages, already handled

            # Handle different message types
            if msg["role"] == "user":
                # User messages are straightforward
                self.history.append({
                    "role": "user",
                    "content": msg["content"]
                })

            elif msg["role"] == "assistant":
                # For assistant messages with tool calls
                if "metadata" in msg and msg["metadata"].get("tool_info"):
                    # This is an assistant message with tool calls
                    tool_info = msg["metadata"]["tool_info"]

                    # Create OpenAI assistant message with tool calls
                    assistant_msg = {
                        "role": "assistant",
                        "content": msg["content"] if isinstance(msg["content"], str) else "",
                        "tool_calls": [{
                            "id": tool_info["id"],
                            "type": "function",
                            "function": {
                                "name": tool_info["name"],
                                "arguments": json.dumps(tool_info["arguments"]) if isinstance(tool_info["arguments"], dict) else tool_info["arguments"]
                            }
                        }]
                    }
                    self.history.append(assistant_msg)
                else:
                    # Regular assistant message
                    self.history.append({
                        "role": "assistant",
                        "content": msg["content"]
                    })

            elif msg["role"] == "tool":
                # Tool response messages
                if "metadata" in msg and "tool_info" in msg["metadata"]:
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": msg["metadata"]["tool_info"]["id"],
                        "content": json.dumps(msg["content"]) if isinstance(msg["content"], (dict, list)) else msg["content"]
                    }
                    self.history.append(tool_msg)


    def _anthropic_normalizer(self):
        """
        Convert standardized session_history to Anthropic-compatible format in self.history.
        """
        # For Anthropic, we don't include system message in the history
        # but need to handle content blocks for tool use/results
        
        # Start with empty history
        self.history = []

        # Process all non-system messages
        for msg in self.session_history:
            if msg["role"] == "system":
                # Update system prompt if this is the most recent system message
                # (only apply the most recent system message if we have multiple)
                if msg == self.session_history[-1] or all(m["role"] != "system" for m in self.session_history[self.session_history.index(msg)+1:]):
                    self.system = msg["content"]
                continue  # Skip system messages in history

            # Handle different message types
            if msg["role"] == "user":
                # User messages - check if it contains tool results
                if "metadata" in msg and "tool_info" in msg["metadata"] and msg["metadata"]["tool_info"].get("result"):
                    # This is a tool result message
                    tool_info = msg["metadata"]["tool_info"]

                    # Create Anthropic tool result format
                    tool_result_msg = {
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": tool_info["id"],
                            "content": json.dumps(tool_info["result"]) if isinstance(tool_info["result"], (dict, list)) else str(tool_info["result"])
                        }]
                    }
                    self.history.append(tool_result_msg)
                else:
                    # Regular user message
                    self.history.append({
                        "role": "user",
                        "content": msg["content"]
                    })
            
            elif msg["role"] == "assistant":
                # For assistant messages, check for tool use
                if "metadata" in msg and "tool_info" in msg["metadata"]:
                    # This is an assistant message with tool use
                    tool_info = msg["metadata"]["tool_info"]
                    current_tool_use_id = tool_info["id"]

                    # Build content blocks
                    content_blocks = []

                    # Add text content if present
                    if msg["content"]:
                        content_blocks.append({
                            "type": "text",
                            "text": msg["content"] if isinstance(msg["content"], str) else ""
                        })

                    # Add tool use block
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tool_info["id"],
                        "name": tool_info["name"],
                        "input": tool_info["arguments"] if isinstance(tool_info["arguments"], dict) else json.loads(tool_info["arguments"])
                    })

                    # Create Anthropic assistant message with tool use
                    self.history.append({
                        "role": "assistant",
                        "content": content_blocks
                    })
                else:
                    # Regular assistant message
                    self.history.append({
                        "role": "assistant",
                        "content": msg["content"]
                    })

            elif msg["role"] == "tool":
                # Tool messages in standard format get converted to user messages with tool_result
                if "metadata" in msg and "tool_info" in msg["metadata"]:
                    tool_info = msg["metadata"]["tool_info"]

                    # Create Anthropic tool result message
                    tool_result_msg = {
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": tool_info["id"],
                            "content": json.dumps(msg["content"]) if isinstance(msg["content"], (dict, list)) else str(msg["content"])
                        }]
                    }
                    self.history.append(tool_result_msg)


    def _openai_normalize_message(self, msg):
        """Normalize a single message to OpenAI format and add to history."""
        role = msg.get("role")
        content = msg.get("content")

        if role == "system":
            # Check if we already have a system message in history
            system_index = next((i for i, m in enumerate(self.history)
                                 if m.get("role") == "system"), None)
            if system_index is not None:
                # Update existing system message
                self.history[system_index]["content"] = content
            else:
                # Insert new system message at the beginning
                self.history.insert(0, {
                    "role": "system",
                    "content": content
                })
            # Update the system property
            self.system = content

        elif role == "user":
            self.history.append({
                "role": "user",
                "content": content
            })

        elif role == "assistant":
            # For assistant messages, handle potential tool calls
            if "metadata" in msg and msg["metadata"].get("tool_info"):
                # This is an assistant message with tool calls
                tool_info = msg["metadata"]["tool_info"]
                
                # Create OpenAI assistant message with tool calls
                try:
                    arguments = tool_info.get("arguments", {})
                    arguments_str = json.dumps(arguments) if isinstance(arguments, dict) else arguments
                except:
                    arguments_str = str(arguments)
                    
                assistant_msg = {
                    "role": "assistant",
                    "content": content if isinstance(content, str) else "",
                    "tool_calls": [{
                        "id": tool_info["id"],
                        "type": "function",
                        "function": {
                            "name": tool_info["name"],
                            "arguments": arguments_str
                        }
                    }]
                }
                self.history.append(assistant_msg)
            else:
                # Regular assistant message
                self.history.append({
                    "role": "assistant",
                    "content": content
                })
            
        elif role == "tool":
            # Tool response messages
            if "metadata" in msg and "tool_info" in msg["metadata"]:
                tool_info = msg["metadata"]["tool_info"]
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tool_info["id"],
                    "content": json.dumps(content) if isinstance(content, (dict, list)) else str(content)
                }
                self.history.append(tool_msg)


    def _anthropic_normalize_message(self, msg):
        """Normalize a single message to Anthropic format and add to history."""
        role = msg.get("role")
        content = msg.get("content")

        if role == "system":
            # Store system prompt separately, not in history for Anthropic
            self.system = content

        elif role == "user":
            # User messages - check if it contains tool results
            if "metadata" in msg and "tool_info" in msg["metadata"]:
                tool_info = msg["metadata"]["tool_info"]
                # Check for result or directly use content
                result_content = tool_info.get("result", content)

                # Create Anthropic tool result format
                try:
                    result_str = json.dumps(result_content) if isinstance(result_content, (dict, list)) else str(result_content)
                except:
                    result_str = str(result_content)

                tool_result_msg = {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_info["id"],
                        "content": result_str
                    }]
                }
                self.history.append(tool_result_msg)
            else:
                # Regular user message
                self.history.append({
                    "role": "user",
                    "content": content
                })

        elif role == "assistant":
            # For assistant messages, check for tool use
            if "metadata" in msg and "tool_info" in msg["metadata"]:
                # This is an assistant message with tool use
                tool_info = msg["metadata"]["tool_info"]

                # Build content blocks
                content_blocks = []

                # Add text content if present
                if content:
                    content_blocks.append({
                        "type": "text",
                        "text": content if isinstance(content, str) else ""
                    })

                # Add tool use block - safely convert arguments
                try:
                    # Parse arguments to ensure it's a dictionary
                    if isinstance(tool_info["arguments"], str):
                        try:
                            args_dict = json.loads(tool_info["arguments"])
                        except json.JSONDecodeError:
                            args_dict = {"text": tool_info["arguments"]}
                    else:
                        args_dict = tool_info["arguments"]
                except:
                    args_dict = {"error": "Failed to parse arguments"}

                content_blocks.append({
                    "type": "tool_use",
                    "id": tool_info["id"],
                    "name": tool_info["name"],
                    "input": args_dict
                })

                # Create Anthropic assistant message with tool use
                self.history.append({
                    "role": "assistant",
                    "content": content_blocks
                })
            else:
                # Regular assistant message
                self.history.append({
                    "role": "assistant",
                    "content": content
                })

        elif role == "tool":
            # Tool messages in standard format get converted to user messages with tool_result
            if "metadata" in msg and "tool_info" in msg["metadata"]:
                tool_info = msg["metadata"]["tool_info"]

                try:
                    result_str = json.dumps(content) if isinstance(content, (dict, list)) else str(content)
                except:
                    result_str = str(content)

                # Create Anthropic tool result message
                tool_result_msg = {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_info["id"],
                        "content": result_str
                    }]
                }
                self.history.append(tool_result_msg)


    def _generic_normalize_message(self, msg):
        """Generic normalizer for unknown SDKs."""
        role = msg.get("role")
        content = msg.get("content")
        
        if role in ["user", "assistant", "system"]:
            self.history.append({
                "role": role,
                "content": content
            })


    def track_token_usage(self):
        """Track and return token usage across the conversation history.
        
        Returns:
            dict: Dictionary containing current token counts, limits, and history.
        """
        if not hasattr(self, "_token_history"):
            self._token_history = []
        
        # Count current tokens
        current_count = self._count_tokens(self.history)
        
        # Add to history
        timestamp = datetime.now(timezone.utc).isoformat()
        self._token_history.append({
            "timestamp": timestamp,
            "count": current_count,
            "limit": self.context_length,
            "provider": self.provider,
            "model": self.model
        })
        
        # Keep only the last 100 measurements to avoid unlimited growth
        if len(self._token_history) > 100:
            self._token_history = self._token_history[-100:]
        
        # Return current tracking info
        return {
            "current": current_count,
            "limit": self.context_length,
            "percentage": round((current_count / self.context_length) * 100, 1) if self.context_length else 0,
            "history": self._token_history[-10:],  # Return last 10 measurements
            "provider": self.provider,
            "model": self.model
        }


    def get_message_token_breakdown(self):
        """Analyze token usage by message type and provide a detailed breakdown.
        
        Returns:
            dict: Token usage broken down by message types and roles.
        """
        breakdown = {
            "total": 0,
            "by_role": {
                "system": 0,
                "user": 0,
                "assistant": 0,
                "tool": 0
            },
            "by_type": {
                "text": 0,
                "tool_calls": 0,
                "tool_results": 0
            },
            "messages": []
        }
        
        # Analyze each message
        for i, msg in enumerate(self.history):
            msg_tokens = self._count_tokens([msg])
            role = msg.get("role", "unknown")
            
            # Track by role
            if role in breakdown["by_role"]:
                breakdown["by_role"][role] += msg_tokens
            
            # Track by content type
            if role == "assistant" and msg.get("tool_calls"):
                breakdown["by_type"]["tool_calls"] += msg_tokens
            elif role == "tool":
                breakdown["by_type"]["tool_results"] += msg_tokens
            else:
                breakdown["by_type"]["text"] += msg_tokens
            
            # Add individual message data
            breakdown["messages"].append({
                "index": i,
                "role": role,
                "tokens": msg_tokens,
                "has_tools": bool(msg.get("tool_calls") or msg.get("tool_use") or 
                                (isinstance(msg.get("content"), list) and 
                                 any(isinstance(c, dict) and c.get("type") in ["tool_use", "tool_result"] 
                                     for c in msg.get("content", []))))
            })
            
            # Update tota?
            breakdown["total"] += msg_tokens
        
        return breakdown


def run_bash_command(command: str, safe_mode: bool = True) -> Dict[str, Any]:
    """Run a simple bash command (e.g., 'ls -la ./' to list files) and return the output.

    Args:
        command: Shell command to execute.
        safe_mode: If True, asks for confirmation before execution.

    Returns:
        Dict containing status, output, error, and return_code.
    """
    print(Syntax(f"\n{command}\n", "bash", theme="monokai"))

    if safe_mode:
        if not Confirm.ask("execute? [y/n]: ", default=False):
            return {"status": "cancelled"}

    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=10)
        print(Rule(), result.stdout.strip(), Rule())
        return {
            "status": "success",
            "output": result.stdout.strip(),
            "error": result.stderr.strip() or None,
            "return_code": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": "Command timed out."}
    except Exception as e:
        return {"status": "error", "error": str(e)}
    

def main():
    """
    Run the Interactor as a standalone AI chat client via CLI.

    This function sets up argument parsing, initializes the Interactor, registers tools,
    and enters a user interaction loop. It supports streaming, markdown, and tool calling
    via command-line flags.

    Command-line arguments:
        --model:      Model identifier in "provider:model" format (default: openai:gpt-4o-mini)
        --base-url:   Optional override for the model's base URL
        --api-key:    Optional API key override
        --stream:     Enable streaming response output
        --markdown:   Enable markdown formatting in the terminal
        --tools:      Enable function/tool calling

    Returns:
        int: Exit code. 0 on normal termination, 1 if initialization fails.

    Raises:
        SystemExit: On argparse errors or manual termination (e.g. KeyboardInterrupt).
    """
    parser = argparse.ArgumentParser(description='AI Chat Client')
    parser.add_argument('--model', default='openai:gpt-4o-mini',
                        help='Model identifier in format "provider:model_name"')
    parser.add_argument('--base-url', help='Base URL for API (optional)')
    parser.add_argument('--api-key', help='API key (optional)')
    parser.add_argument('--stream', action='store_true', default=True,
                        help='Enable response streaming (default: True)')
    parser.add_argument('--markdown', action='store_true', default=False,
                        help='Enable markdown rendering (default: False)')
    parser.add_argument('--tools', action='store_true', default=True,
                        help='Enable tool calling (default: True)')

    args = parser.parse_args()

    try:
        caller = Interactor(
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            tools=args.tools,
            stream=args.stream,
            context_length=500
        )

        caller.add_function(run_bash_command)

        caller.system = caller.messages_system(
            "You are a helpful assistant. Only call tools if one is applicable."
        )

        print("[bold green]Interactor Class[/bold green]")

        while True:
            try:
                user_input = input("\nYou: ").strip()

                if user_input.lower() in {"/exit", "/quit"}:
                    break
                if user_input.startswith("/list"):
                    models = caller.list_models()
                    print(models)
                    continue
                elif not user_input:
                    continue

                response = caller.interact(
                    user_input,
                    tools=args.tools,
                    stream=args.stream,
                    markdown=args.markdown
                )

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error in main loop: {e}")
                continue

    except Exception as e:
        print(f"[red]Failed to initialize chat client: {str(e)}[/red]")
        return 1

    return 0

if __name__ == "__main__":
    main()
