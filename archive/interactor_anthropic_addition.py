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
# Modified: 2025-05-11 16:47:27

import os
import re
import sys
import json
import subprocess
import inspect
import argparse
import tiktoken
import asyncio
import aiohttp
import logging
import traceback

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
from datetime import datetime

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
        context_length: int = 128000,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        log_path: Optional[str] = None,
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
        self.logger = logging.getLogger(f"InteractorLogger_{id(self)}")
        self.logger.setLevel(logging.DEBUG)

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
        }
        """
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
        """

        if model is None:
            model = "openai:gpt-4o-mini"

        # Initialize model + encoding
        self.system = self.messages_system("You are a helpful Assistant.")
        self._setup_client(model, base_url, api_key)
        self.tools_enabled = self.tools_supported if tools is None else tools and self.tools_supported
        self._setup_encoding()

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
        self._normalize_history_to_sdk()

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
        disabled: bool = False
    ):
        """
        Register a function for LLM tool calling with full type hints and metadata.

        Args:
            external_callable (Callable): The function to register.
            name (Optional[str]): Optional custom name. Defaults to function's __name__.
            description (Optional[str]): Optional custom description. Defaults to first line of docstring.
            override (bool): If True, replaces an existing tool with the same name.
            disabled (bool): If True, registers the function in a disabled state.

        Raises:
            ValueError: If the callable is invalid or duplicate name found without override.

        Example:
            interactor.add_function(my_tool, override=True, disabled=False)
        """
        def _python_type_to_schema(ptype: Any) -> dict:
            """Convert a Python type annotation to OpenAI-compatible JSON Schema."""
            origin = get_origin(ptype)
            args = get_args(ptype)

            if origin is Union and type(None) in args:
                non_none = [a for a in args if a is not type(None)]
                if len(non_none) == 1:
                    inner = _python_type_to_schema(non_none[0])
                    return {**inner, "nullable": True}
                return {"type": "object"}  # fallback

            if origin in (list, List):
                item_type = args[0] if args else str
                return {"type": "array", "items": _python_type_to_schema(item_type)}
            if origin in (dict, Dict):
                return {"type": "object"}  # optionally expand props later
            if ptype == str:
                return {"type": "string"}
            if ptype in (int, float):
                return {"type": "number"}
            if ptype == bool:
                return {"type": "boolean"}

            return {"type": "object"}

        def _parse_param_docs(docstring: str) -> dict:
            """Extract parameter descriptions from a docstring."""
            if not docstring:
                return {}

            lines = docstring.splitlines()
            param_docs = {}
            current_param = None
            in_params = False

            param_section_re = re.compile(r"^(Args|Parameters):\s*$")
            param_line_re = re.compile(r"^\s{4}(\w+)\s*(?:\([^\)]*\))?:\s*(.*)")

            for line in lines:
                if param_section_re.match(line.strip()):
                    in_params = True
                    continue
                if in_params:
                    if not line.strip():
                        continue
                    match = param_line_re.match(line)
                    if match:
                        current_param = match.group(1)
                        param_docs[current_param] = match.group(2).strip()
                    elif current_param and line.startswith(" " * 8):
                        param_docs[current_param] += " " + line.strip()
                    else:
                        current_param = None  # reset on malformed or unrelated line

            return param_docs

        if not self.tools_enabled:
            return
        if not external_callable:
            raise ValueError("A valid external callable must be provided.")

        function_name = name or external_callable.__name__
        docstring = inspect.getdoc(external_callable)
        description = description or (docstring.split("\n")[0] if docstring else "No description provided.")
        param_docs = _parse_param_docs(docstring)

        if override:
            self.delete_function(function_name)
        elif any(t["function"]["name"] == function_name for t in self.tools):
            raise ValueError(f"Function '{function_name}' is already registered. Use override=True to replace.")

        signature = inspect.signature(external_callable)
        properties = {}
        required = []

        for param_name, param in signature.parameters.items():
            schema = _python_type_to_schema(param.annotation)
            schema["description"] = param_docs.get(param_name, f"{param_name} parameter")
            properties[param_name] = schema

            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        tool_spec = {
            "type": "function",
            "function": {
                "name": function_name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }

        if disabled:
            tool_spec["function"]["disabled"] = True

        self.tools.append(tool_spec)
        setattr(self, function_name, external_callable)

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
                        self._normalize_history_to_sdk()
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
        session_id: Optional[str] = None
    ) -> Optional[str]:
        """Main universal gateway for all LLM interaction.
        
        This method handles the complete interaction pipeline, including:
        - Token estimation and context management
        - Tool calling functionality with proper looping
        - Streaming for the complete interaction (including after tool calls)
        - Session management
        
        Args:
            user_input: The user's message
            quiet: If True, suppresses console output
            tools: If True, enables tool calling
            stream: If True, streams the response tokens
            markdown: If True, renders responses as markdown
            model: Optional model override
            output_callback: Optional callback for streaming tokens
            session_id: Optional session ID for history persistence
            
        Returns:
            The complete response text, or None if the input was empty
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
        user_msg = {"role": "user", "content": user_input}
        self.history.append(user_msg)

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

        # Save user message to session
        if self.session_enabled and self.session_id:
            self.session.msg_insert(self.session_id, user_msg)

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
        
        # Use the same streaming settings for all iterations if streaming is enabled
        stream_all = stream
        
        # Main interaction loop - continues until no more tool calls or max iterations reached
        while iterations < max_iterations:
            iterations += 1
            
            try:
                # Execute the appropriate SDK runner
                response_data = await self.sdk_runner(
                    model=self.model,
                    messages=self.history,
                    stream=stream_all,  # Use the same streaming setting for all iterations
                    markdown=markdown,
                    quiet=quiet if iterations == 1 else False,  # Only be quiet on first iteration if requested
                    live=live,
                    output_callback=output_callback
                )
                
                # Extract response data
                content = response_data.get("content", "")
                tool_calls = response_data.get("tool_calls", [])
                
                # Log the response data for debugging
                self._log(f"[ITERATION {iterations}] Content: {len(content)} chars, Tool calls: {len(tool_calls)}")
                
                # Add content to full response - for first response or continuations
                if iterations == 1:
                    full_content = content
                elif content:
                    # For continuations, make it clear this is a continuation after tool usage
                    if not stream_all and not quiet:
                        print(f"\n[Model continuation after tool call]: {content}")
                    
                    # Only add a separator if both have content
                    if full_content and content:
                        full_content += f"\n{content}"
                    else:
                        full_content = content
                
                # Create assistant message for history
                if len(tool_calls) > 0:
                    # With tool calls, create different assistant messages based on SDK
                    if self.sdk == "openai":
                        # Format for OpenAI
                        assistant_msg = {
                            "role": "assistant",
                            "content": content,
                            "tool_calls": [
                                {
                                    "id": call["id"],
                                    "type": "function",
                                    "function": {
                                        "name": call["function"]["name"],
                                        "arguments": call["function"]["arguments"]
                                    }
                                } for call in tool_calls
                            ]
                        }
                        
                    elif self.sdk == "anthropic":
                        # For Anthropic, we need to handle each tool call separately
                        msg_content = []
                        
                        # Add text content if present
                        if content:
                            format_content = {
                                "type": "text",
                                "text": content
                            }
                            msg_content.append(format_content)

                        # Add tool calls as properly formatted objects
                        for call in tool_calls:
                            # Parse arguments to ensure it's a dictionary
                            try:
                                # If arguments is a string, parse it to dictionary
                                if isinstance(call["function"]["arguments"], str):
                                    args_dict = json.loads(call["function"]["arguments"])
                                else:
                                    args_dict = call["function"]["arguments"]
                            except json.JSONDecodeError:
                                # If parsing fails, create a simple text dictionary
                                args_dict = {"text": call["function"]["arguments"]}
                                
                            format_tool_calls = {
                                "type": "tool_use",
                                "id": call["id"],
                                "name": call["function"]["name"],
                                "input": args_dict  # Must be a dictionary, not a string
                            }
                            msg_content.append(format_tool_calls)

                        assistant_msg = {"role": "assistant", "content": msg_content}

                    # Generic Session Recorded for session history
                    session_msg = {
                        "role": "assistant",
                        "content": content, 
                        "tool_calls": tool_calls
                    }
                else:
                    # Simple response without tool calls
                    assistant_msg = {"role": "assistant", "content": content}
                    session_msg = assistant_msg

                self.history.append(assistant_msg) 
                if self.session_enabled and self.session_id:
                    self.session.msg_insert(self.session_id, session_msg)

                # If no tool calls or tools disabled, we're done
                if not tool_calls or not tool_enabled:
                    break
                
                # Process each tool call
                for call in tool_calls:
                    call_name = call["function"]["name"]
                    call_args = call["function"]["arguments"]
                    call_id = call["id"]
                    
                    # Execute the tool
                    result = await self._handle_tool_call_async(
                        call_name, call_args, call_id,
                        {"model": self.model, "messages": self.history, "stream": stream},
                        markdown, live, False, output_callback
                    )
                    
                    # Format the tool result based on SDK
                    if self.sdk == "openai":
                        tool_msg = {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": json.dumps(result) if isinstance(result, (dict, list)) else str(result)
                        }
                    elif self.sdk == "anthropic":
                        # Anthropic uses a special user message with tool_result content
                        tool_msg = {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": call_id,
                                    "content": json.dumps(result) if isinstance(result, (dict, list)) else str(result)
                                }
                            ]
                        }
                    else:
                        # Generic fallback
                        tool_msg = {
                            "role": "tool",
                            "content": json.dumps(result) if isinstance(result, (dict, list)) else str(result),
                            self.tool_key: call_id
                        }
                    
                    # Generic session history format
                    session_tool_msg = {
                        "role": "tool",
                        "content": json.dumps(result) if isinstance(result, (dict, list)) else str(result),
                        self.tool_key: call_id
                    }
                    
                    # Add tool message to history
                    self.history.append(tool_msg)
                    if self.session_enabled and self.session_id:
                        self.session.msg_insert(self.session_id, session_tool_msg)
                    
                    # If we're streaming, indicate that we're waiting for the model's response to the tool call
                # If we're in streaming mode and have Live, we need to reset it for the next iteration
                if stream_all and live:
                    # Stop the current live instance
                    live.stop()
                    # Create a new live instance for the next iteration
                    live = Live(console=console, refresh_per_second=100)
                    live.start()
                    
                # Continue the loop to get the model's response to tool calls    
                
            except Exception as e:
                self.logger.error(f"[{self.sdk.upper()} ERROR] {str(e)}")
                self._log(f"[ERROR] Error in interaction loop: {str(e)}", level="error")
                if live:
                    live.stop()
                return f"Error: {str(e)}"
        
        # Clean up live display if needed
        if live:
            live.stop()
        
        # Ensure we have a reasonable response even if something went wrong
        return full_content or "No response."

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
        # Normalize messages for OpenAI format
        openai_messages = []
        for msg in messages:
            # Skip duplicate system messages
            if msg.get("role") == "system" and any(m.get("role") == "system" for m in openai_messages):
                continue
            
            # Handle special message formats
            if msg.get("role") == "tool":
                # OpenAI requires tool_call_id for tool messages
                tool_msg = {
                    "role": "tool",
                    "content": msg.get("content", ""),
                    "tool_call_id": msg.get("tool_call_id") or msg.get(self.tool_key)
                }
                
                # Skip malformed tool messages
                if not tool_msg.get("tool_call_id"):
                    self._log(f"[OPENAI WARNING] Skipping malformed tool message: {msg}", level="warning")
                    continue
                    
                openai_messages.append(tool_msg)
            elif msg.get("role") == "assistant" and msg.get("tool_calls"):
                # Handle assistant messages with tool calls
                asst_msg = {
                    "role": "assistant",
                    "content": msg.get("content", "")
                }
                
                # Add tool_calls if present
                if msg.get("tool_calls"):
                    asst_msg["tool_calls"] = msg["tool_calls"]
                    
                openai_messages.append(asst_msg)
            elif msg.get("role") == "user" and isinstance(msg.get("content"), list):
                # Convert Anthropic-style user messages with tool results to OpenAI format
                # This detects when a message was created for Anthropic but is now being used with OpenAI
                tool_results = [item for item in msg.get("content", []) if isinstance(item, dict) and item.get("type") == "tool_result"]
                
                if tool_results:
                    for result in tool_results:
                        tool_msg = {
                            "role": "tool",
                            "content": result.get("content", ""),
                            "tool_call_id": result.get("tool_use_id")
                        }
                        if tool_msg.get("tool_call_id"):
                            openai_messages.append(tool_msg)
                else:
                    # Regular user message, convert list content to string if needed
                    user_msg = {
                        "role": "user",
                        "content": msg.get("content", "") if isinstance(msg.get("content"), str) else json.dumps(msg.get("content", ""))
                    }
                    openai_messages.append(user_msg)
            else:
                # Regular message, just copy it
                new_msg = {
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                }
                openai_messages.append(new_msg)
        
        # Log the messages we're sending
        self._log(f"[OPENAI REQUEST] Sending {len(openai_messages)} messages to {model}", level="debug")
        for i, msg in enumerate(openai_messages):
            role = msg.get("role", "unknown")
            has_tool_calls = "tool_calls" in msg
            tool_call_id = msg.get("tool_call_id", "none")
            self._log(f"[OPENAI MESSAGE {i}] role={role}, has_tool_calls={has_tool_calls}, tool_call_id={tool_call_id}", level="debug")

        # Prepare API parameters
        params = {
            "model": model,
            "messages": openai_messages,
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
            self.logger.error(f"[OPENAI ERROR] {str(e)}")
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
                    elif not markdown and not quiet:
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
        """Handle Anthropic-specific API interactions and response processing with enhanced tool calling."""
        # Convert messages to Anthropic format
        anthropic_messages = []
        last_assistant_had_tool_use = False
        system_prompt = self.system
        
        # Check if the last message is from the user and contains certain trigger phrases
        last_message_is_trigger = False
        if messages and messages[-1].get("role") == "user":
            user_content = messages[-1].get("content", "")
        
        # Process messages in order
        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            # Skip system messages - handled separately in Anthropic
            if role == "system":
                system_prompt = content
                continue
            
            # Handle different message types
            if role == "assistant":
                # Check if this is a tool use message
                if msg.get("tool_use") or msg.get("tool_calls"):
                    # For direct tool_use (already in Anthropic format)
                    if msg.get("tool_use"):
                        anthropic_messages.append(msg)
                        last_assistant_had_tool_use = True
                        continue
                    
                    """
                    # For OpenAI-style tool_calls, convert to Anthropic tool_use format
                    for tool_call in msg.get("tool_calls", []):
                        # Extract the data
                        tool_id = tool_call.get("id")
                        name = tool_call.get("function", {}).get("name", "")
                        arguments = tool_call.get("function", {}).get("arguments", "{}")
                        
                        # Convert arguments to proper format
                        if isinstance(arguments, str):
                            try:
                                args_dict = json.loads(arguments)
                            except json.JSONDecodeError:
                                args_dict = {"text": arguments}
                        else:
                            args_dict = arguments
                        
                        # Create the Anthropic tool use message
                        tool_use_msg = {
                            "role": "assistant",
                            "content": "",
                            "tool_use": {
                                "id": tool_id,
                                "name": name,
                                "input": args_dict
                            }
                        }
                        
                        anthropic_messages.append(tool_use_msg)
                        last_assistant_had_tool_use = True
                    """
                    
                    # If the message also had content, add it as a separate message
                    if content:
                        anthropic_messages.append({
                            "role": "assistant", 
                            "content": content
                        })
                    
                    continue
                
                # Regular assistant message (no tool calls)
                if content:
                    anthropic_messages.append({"role": "assistant", "content": content})
                    last_assistant_had_tool_use = False
            
            # Handle tool responses - must be paired with a previous tool_use
            elif role == "tool" or (role == "user" and isinstance(content, list) and any(isinstance(item, dict) and item.get("type") == "tool_result" for item in content)):
                # Already in Anthropic format with tool_result
                if role == "user" and isinstance(content, list):
                    anthropic_messages.append(msg)
                    continue
                
                # Convert from OpenAI tool format to Anthropic format
                tool_id = msg.get("tool_call_id") or msg.get(self.tool_key)
                result_content = content
                
                # Only add if we have a tool ID and the previous message was a tool use
                if tool_id and last_assistant_had_tool_use:
                    anthropic_messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": result_content
                            }
                        ]
                    })
            
            # Regular user message
            elif role == "user":
                if isinstance(content, str):
                    anthropic_messages.append({"role": "user", "content": content})
                elif isinstance(content, list):
                    # This handles user messages with complex content blocks
                    anthropic_messages.append({"role": "user", "content": content})
        
        # Log the messages we're sending
        self._log(f"[ANTHROPIC REQUEST] Sending {len(anthropic_messages)} messages to {model}", level="debug")
        for i, msg in enumerate(anthropic_messages):
            role = msg.get("role", "unknown")
            has_tool_use = "tool_use" in msg
            has_tool_result = isinstance(msg.get("content"), list) and any(isinstance(item, dict) and item.get("type") == "tool_result" for item in msg.get("content", []))
            self._log(f"[ANTHROPIC MESSAGE {i}] role={role}, has_tool_use={has_tool_use}, has_tool_result={has_tool_result}", level="debug")
        
        # Prepare API parameters
        params = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": 8192,
            "system": system_prompt
        }
        # Add tools support if needed
        if self.tools_enabled and self.tools_supported:
            enabled_tools = []
            for tool in self._get_enabled_tools():
                format_tool = {
                    "name": tool["function"]["name"],
                    "description": tool["function"].get("description", ""),
                    "input_schema": tool["function"]["parameters"],
                }
                enabled_tools.append(format_tool)

            params["tools"] = enabled_tools

        assistant_content = ""
        tool_calls_dict = {}
        
        try:
            # Process streaming response
            if stream:
                stream_response = await self._retry_with_backoff(
                    self.async_client.messages.create,
                    stream=True,
                    **params
                )
               
                content_type = None
                async for chunk in stream_response:
                    data = json.loads(chunk.to_json())
                    chunk_type = getattr(chunk, "type", "unknown")
                    stop_reason = getattr(chunk, "stop_reason", None)
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
                            self._log(f"[ANTHROPIC TOOL USE] Found complete tool use: {tool_name}", level="info")
                    # Handle text content
                    if chunk_type == "content_block_delta" and hasattr(chunk.delta, "text"):
                        delta = chunk.delta.text
                        assistant_content += delta
                        if output_callback:
                            output_callback(delta)
                        elif live:
                            live.update(Markdown(assistant_content))
                        elif not markdown and not quiet:
                            print(delta, end="")

                    # Handle complete tool use
                    elif chunk_type == "content_block_delta" and content_type == "tool_use":
                        tool_calls_dict[tool_id]["function"]["arguments"] += chunk.delta.partial_json

            # Process non-streaming response
            else:
                response = await self._retry_with_backoff(
                    self.async_client.messages.create,
                    **params
                )
                
                # Extract text content
                for content_block in response.content:
                    if content_block.type == "text":
                        assistant_content += content_block.text
                
                # Extract tool uses
                tool_uses = getattr(response, "tool_uses", [])
                if tool_uses:
                    self._log(f"[ANTHROPIC TOOL USES] Found {len(tool_uses)} tool uses", level="debug")
                    for i, tool_use in enumerate(tool_uses):
                        # Extract and format the tool use
                        tool_id = tool_use.id
                        tool_name = tool_use.name
                        tool_input = tool_use.input
                        
                        # Format the input as JSON string
                        if isinstance(tool_input, dict):
                            input_json = json.dumps(tool_input)
                        else:
                            input_json = json.dumps({}) if tool_input is None else str(tool_input)
                        
                        tool_calls_dict[tool_id] = {
                            "id": tool_id,
                            "function": {
                                "name": tool_name,
                                "arguments": input_json
                            }
                        }
                        self._log(f"[ANTHROPIC TOOL USE] {tool_name}", level="debug")
                
                if output_callback:
                    output_callback(assistant_content)
                elif not quiet:
                    print(assistant_content)
                    
        except Exception as e:
            self.logger.error(f"Error in Anthropic runner: {traceback.print_exc()}")
            # Add more detailed logging
            self._log(f"[ANTHROPIC ERROR] {traceback.format_exc()}", level="error")
            
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
        params: dict,
        markdown: bool,
        live: Optional[Live],
        safe: bool = False,
        output_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """Process a tool call asynchronously and return the result.

        Args:
            function_name: Name of the function to call.
            function_arguments: JSON string containing the function arguments.
            tool_call_id: Unique identifier for this tool call.
            params: Parameters used for the original API call.
            markdown: If True, renders content as markdown.
            live: Optional Live context for updating content in real-time.
            safe: If True, prompts for confirmation before executing the tool call.
            output_callback: Optional callback to handle the tool call result.

        Returns:
            The result of the function call.

        Raises:
            ValueError: If the function is not found or JSON is invalid.
        """
        live_was_active = False
        
        try:
            arguments = json.loads(function_arguments or "{}")
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in tool call arguments: {e}")
            return {"error": "Invalid JSON format in tool call arguments."}

        self._log(f"[TOOL:{function_name}] args={arguments}")

        func = getattr(self, function_name, None)
        if not func:
            raise ValueError(f"Function '{function_name}' not found.")

        # Properly handle live display if it's active
        if live and live.is_started:
            live_was_active = True
            live.stop()

        print(f"Running {function_name}...\n")

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

            # Restart live display if it was active before
            if live_was_active and live:
                live.start()

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

    def _count_tokens(self, messages) -> int:
        """Accurately estimate token count for messages including tool calls.

        Args:
            messages: List of message objects in either OpenAI or Anthropic format.

        Returns:
            int: Estimated token count.
        """
        if not hasattr(self, "encoding") or not self.encoding:
            self._setup_encoding()

        # For Claude models, try to use their built-in token counter if available
        if self.sdk == "anthropic":
            try:
                # Convert messages to Anthropic format if needed
                anthropic_messages = []
                for msg in messages:
                    if msg.get("role") == "system":
                        continue  # System handled separately

                    if msg.get("role") == "tool":
                        # Skip tool messages in token count to avoid double-counting
                        # since the Anthropic token counter expects tool_result format
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

                # Use Anthropic's token counter
                response = self.client.messages.count_tokens(
                    model=self.model,
                    messages=anthropic_messages,
                    system=self.system
                )
                return response.input_tokens
            except Exception as e:
                # Fall back to our estimation
                self._log(f"[TOKEN COUNT] Error using Anthropic token counter: {e}", level="warning")
                pass

        # Fallback counting method
        num_tokens = 0
        for msg in messages:
            # Base token count for message metadata
            num_tokens += 4  # Message overhead

            # Count tokens in message content
            if isinstance(msg.get("content"), str):
                content = msg.get("content", "")
                num_tokens += len(self.encoding.encode(content))
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
                                num_tokens += len(self.encoding.encode(json.dumps(result_content)))
                            # Add tokens for tool_use_id and type fields
                            num_tokens += 10
                    else:
                        # Plain text content
                        num_tokens += len(self.encoding.encode(str(item)))

            # Count tokens in tool calls
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

                        # Add tokens for id and other metadata
                        num_tokens += 10

            # Count tokens in Anthropic tool_use
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

                # Add tokens for id and other metadata
                num_tokens += 10

            # Add tokens for role name
            num_tokens += len(self.encoding.encode(msg.get("role", "")))

        # Add message end tokens
        num_tokens += 2

        return num_tokens


    def _cycle_messages(self):
        """Intelligently trim the message history to fit within the allowed context length.

        This method removes older messages as needed while preserving context coherence.
        It prioritizes keeping system messages and recent messages while gradually removing
        older exchanges.

        Returns:
            bool: True if all messages were trimmed (context exceeded), False otherwise.
        """
        # Check if we need to trim
        token_count = self._count_tokens(self.history)
        if token_count <= self.context_length:
            return False
        
        exceeded_context = False
        messages_removed = 0
        
        # Define a rough token per message estimate to avoid recounting every loop
        avg_tokens_per_msg = max(1, token_count // max(1, len(self.history)))
        
        # Try to estimate how many messages to remove
        messages_to_trim = max(1, (token_count - self.context_length) // avg_tokens_per_msg)
        
        # Keep track of important messages that shouldn't be removed
        kept_indices = []
        
        # Always keep the system message (index 0 if it exists)
        if self.history and self.history[0].get("role") == "system":
            kept_indices.append(0)
        
        # Always keep the latest user message and any subsequent messages
        if len(self.history) >= 2:
            for i in range(len(self.history) - 1, -1, -1):
                if self.history[i].get("role") == "user":
                    # Keep this user message and all messages after it
                    kept_indices.extend(range(i, len(self.history)))
                    break
        
        # Make a copy of the history for trimming
        trimmed_history = [msg for i, msg in enumerate(self.history) if i in kept_indices]
        
        # Now remove oldest non-preserved messages until we're under the token limit
        remove_candidates = [i for i in range(len(self.history)) if i not in kept_indices]
        remove_candidates.sort()  # Remove oldest messages first
        
        removed = 0
        for i in remove_candidates:
            if self._count_tokens(trimmed_history) <= self.context_length:
                break
                
            self._log(f"[TRIM] Removed message: {self.history[i]}", level="debug")
            removed += 1
            messages_removed += 1
        
        # Update history with trimmed version
        self.history = [msg for i, msg in enumerate(self.history) if i in kept_indices or i not in remove_candidates[:removed]]
        
        # If still over the limit, more aggressive trimming
        while self._count_tokens(self.history) > self.context_length and self.history:
            # Find the oldest non-system message to remove
            for i in range(len(self.history)):
                if self.history[i].get("role") != "system":
                    self._log(f"[TRIM] Emergency removed message: {self.history[i]}", level="debug")
                    self.history.pop(i)
                    messages_removed += 1
                    break
        
        # Check if we removed all messages
        if not self.history or (len(self.history) == 1 and self.history[0].get("role") == "system"):
            self._log(f"[TRIM] Context length exceeded - all interactive messages trimmed", level="error")
            exceeded_context = True
        
        if messages_removed > 0:
            self._log(f"[TRIM] Removed {messages_removed} messages to fit within context limit ({self.context_length})", level="info")
        
        return exceeded_context

    def messages_add(
        self,
        role: Optional[str] = None,
        content: Optional[str] = None
    ) -> list:
        """Manage messages in the conversation history."""
        if role is None and content is None:
            return self.history

        if content is None and role is not None:
            raise ValueError("Content must be provided when role is specified")
        if not content:
            raise ValueError("Content cannot be empty")
        if not isinstance(content, str):
            raise ValueError("Content must be a string")

        if role == "system":
            self.messages_system(content)
            return self.history

        if role is not None:
            message = {"role": role, "content": content}
            self.history.append(message)
            if self.session_enabled and self.session_id:
                self.session.msg_insert(self.session_id, message)

            self._log(f"[{role.upper()}] {content}")
            return self.history

        return self.history

    def messages_system(self, prompt: str):
        """Set or retrieve the current system prompt."""
        if not isinstance(prompt, str) or not prompt:
            return self.system

        self.system = prompt

        # Inject system prompt into history if using OpenAI
        if self.sdk == "openai":
            self.history.insert(0, {"role": "system", "content": prompt})

        # Always log it to session
        if self.session_enabled and self.session_id:
            self.session.msg_insert(self.session_id, {"role": "system", "content": prompt})

        return self.system

    def messages(self) -> list:
        """Return full session messages (persisted or in-memory)."""
        if self.session_enabled and self.session_id:
            return self.session.load_full(self.session_id).get("messages", [])
        return self.history

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
        """
        Load and normalize messages for a specific session.

        This activates session persistence, restores the prior conversation state,
        and normalizes message format to match the active SDK (OpenAI/Anthropic/etc).

        Args:
            session_id (Optional[str]): The session ID to activate, or None for in-memory mode.
        """
        self.session_id = session_id
        self._last_session_id = session_id

        if self.session_enabled and session_id:
            try:
                self.history = self.session.load(session_id)
                self._log(f"[SESSION] Switched to session '{session_id}'")
            except Exception as e:
                self.logger.error(f"Failed to load session '{session_id}': {e}")
                self.history = []
        else:
            self.history = []

        self._normalize_history_to_sdk()


    def session_reset(self):
        """
        Reset the current session state and reinitialize to default system prompt.

        Clears history, disables session ID tracking, and returns to in-memory mode.
        """
        self.session_id = None
        self._last_session_id = None
        self.history = []
        self.system = self.messages_system("You are a helpful Assistant.")
        self._log("[SESSION] Reset to in-memory mode")

    def _normalize_history_to_sdk(self):
        """Ensure self.history and self.system are compatible with the active SDK."""

        # Remove any duplicate system messages first
        self.history = [m for i, m in enumerate(self.history)
                        if m.get("role") != "system" or
                        all(n.get("role") != "system" for n in self.history[:i])]

        if self.sdk == "anthropic":
            # Strip system role from history
            self.history = [m for m in self.history if m.get("role") != "system"]

        elif self.sdk == "openai":
            # Ensure system prompt is in history (only if not already there)
            if self.system and not any(m.get("role") == "system" for m in self.history):
                self.history.insert(0, {"role": "system", "content": self.system})

        else:
            self.logger.warning(f"[NORMALIZATION] No system handling logic for SDK '{self.sdk}'")

    def _normalizer(self, sdk=None, full_normalize=False, messages=None):
        """
        Central message normalization orchestrator that delegates to SDK-specific normalizers.

        This method determines which SDK-specific normalizer to use and manages when
        normalization should occur.

        Args:
            sdk (str, optional): Target SDK to normalize to. Defaults to self.sdk.
            full_normalize (bool): If True, forces normalization even if SDK hasn't changed.
            messages (list, optional): Specific messages to normalize instead of self.history.
                                      Returns normalized messages without changing self.history.

        Returns:
            list: Normalized messages if 'messages' parameter was provided,
                  otherwise None (and self.history is modified in-place).
        """
        # Determine target SDK
        target_sdk = sdk or self.sdk

        # Check if SDK-specific normalizer exists
        normalizer_method = getattr(self, f"_{target_sdk}_normalizer", None)
        if not normalizer_method:
            self.logger.warning(f"No normalizer defined for SDK '{target_sdk}'. Messages may not be properly formatted.")
            # Return messages as-is if no normalizer exists
            return messages if messages is not None else None

        # If specific messages provided, normalize and return them without changing self.history
        if messages is not None:
            return normalizer_method(messages)

        # Skip normalization if SDK hasn't changed and full normalize not requested
        if hasattr(self, '_last_normalized_sdk') and self._last_normalized_sdk == target_sdk and not full_normalize:
            return

        # Store current SDK for future comparisons
        self._last_normalized_sdk = target_sdk

        # Normalize history in place
        self.history = normalizer_method(self.history)
        self._log(f"[NORMALIZER] Normalized message history to {target_sdk} format ({len(self.history)} messages)")

        return None

    def _openai_normalizer(self, messages):
        """
        Normalize messages to OpenAI API format.

        Converts various message formats (including Anthropic's) to OpenAI's expected format.

        Args:
            messages (list): Messages to normalize

        Returns:
            list: Messages normalized to OpenAI format
        """
        # Create a new list for normalized messages
        normalized = []

        # Handle system messages first
        system_content = self.system
        system_msg_found = False

        # First pass - extract system message and remove duplicates
        for msg in messages:
            if msg.get("role") == "system":
                if not system_msg_found:
                    system_content = msg.get("content", system_content)
                    system_msg_found = True
                # Skip system messages in this pass
                continue
            normalized.append(msg)

        # Start with system message if available
        result = []
        if system_content:
            result.append({"role": "system", "content": system_content})

        # Process each message for OpenAI format
        for msg in normalized:
            role = msg.get("role", "")
            content = msg.get("content", "")

            # Handle Anthropic-style assistant messages with content blocks
            if role == "assistant" and isinstance(content, list):
                text_content = ""
                tool_calls = []

                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_content += block.get("text", "")
                        elif block.get("type") == "tool_use":
                            # Convert Anthropic tool_use to OpenAI tool_calls
                            tool_id = block.get("id")
                            name = block.get("name", "")
                            input_data = block.get("input", {})

                            # Format arguments as JSON string for OpenAI
                            if isinstance(input_data, dict):
                                args_str = json.dumps(input_data)
                            else:
                                args_str = str(input_data)

                            tool_calls.append({
                                "id": tool_id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": args_str
                                }
                            })

                # Create OpenAI assistant message
                openai_msg = {
                    "role": "assistant",
                    "content": text_content
                }

                if tool_calls:
                    openai_msg["tool_calls"] = tool_calls

                result.append(openai_msg)

            # Handle Anthropic-style user messages with tool results
            elif role == "user" and isinstance(content, list):
                tool_results = [item for item in content if isinstance(item, dict) and item.get("type") == "tool_result"]

                if tool_results:
                    # Create separate tool messages for each tool result
                    for tool_result in tool_results:
                        tool_msg = {
                            "role": "tool",
                            "content": tool_result.get("content", ""),
                            "tool_call_id": tool_result.get("tool_use_id")
                        }
                        result.append(tool_msg)
                else:
                    # For non-tool user content, convert to string if needed
                    text_content = ""
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_content += item.get("text", "")
                        elif isinstance(item, str):
                            text_content += item

                    if text_content:
                        result.append({
                            "role": "user",
                            "content": text_content
                        })

            # Handle OpenAI-specific messages (already correctly formatted)
            elif role == "assistant" and msg.get("tool_calls"):
                # Already in OpenAI format with tool calls
                result.append(msg)

            elif role == "tool" and msg.get("tool_call_id"):
                # Already in OpenAI format as tool response
                result.append(msg)

            # Regular messages that don't need special handling
            else:
                # Convert to string content if necessary
                if not isinstance(content, str) and content is not None:
                    content = json.dumps(content)

                result.append({
                    "role": role,
                    "content": content
                })

        return result



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
