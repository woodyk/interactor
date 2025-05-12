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
# Modified: 2025-05-10 23:22:46

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
                # Anthropic currently requires explicit tool definition on call,
                # so we assume if model accepts tools param, it's supported.
                try:
                    _ = self.client.messages.create(
                        model=self.model,
                        messages=[{"role": "user", "content": "ping"}],
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
                    if "tool" in str(e).lower() or "not supported" in str(e).lower():
                        return False
                    raise
                except Exception:
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
                    print(response)
                    for model in response.data:
                        model_id = f"{provider_name}:{model.id}"
                        if not regex_pattern or regex_pattern.search(model_id):
                            models.append(model_id)

                elif sdk == "anthropic":
                    client = Anthropic(api_key=api_key)
                    response = client.models.list()
                    print(response)
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
        """Main universal gateway for all LLM interaction."""
        if not user_input:
            return None

        self._setup_client(model or f"{self.provider}:{self.model}")
        self._setup_encoding()

        if self.session_enabled and session_id:
            self.session_id = session_id
            self.session_load(session_id)

        user_msg = {"role": "user", "content": user_input}
        self.history.append(user_msg)

        self.token_estimate = self._count_tokens(self.history)
        self.last_token_estimate = self.token_estimate
        if not quiet:
            print(f"[dim]Estimated tokens in context: {self.token_estimate} / {self.context_length}[/dim]")

        if self._cycle_messages():
            if not quiet:
                print("[red]Context window exceeded. Cannot proceed.[/red]")
            return None

        if self.session_enabled and self.session_id:
            self.session.msg_insert(self.session_id, user_msg)

        return asyncio.run(self._interact_async_core(
            user_input=user_input,
            quiet=quiet,
            tools=tools,
            stream=stream,
            markdown=markdown,
            output_callback=output_callback
        ))

    async def _interact_async_core(
        self,
        user_input: str,
        quiet: bool = False,
        tools: bool = True,
        stream: bool = True,
        markdown: bool = False,
        output_callback: Optional[Callable] = None
    ) -> str:
        """Main SDK-agnostic async execution pipeline."""
        messages = self.history.copy()
        model_id = f"{self.provider}:{self.model}"

        kwargs = {
            "model": self.model,
            "messages": messages,
            "stream": stream
        }

        try:
            response = await self.sdk_runner(**kwargs)
        except Exception as e:
            self.logger.error(f"[{self.sdk.upper()} ERROR] {e}")
            return "Model error."

        assistant_content = ""
        tool_calls = []

        # Streaming logic
        if stream and hasattr(response, "__aiter__"):
            live = Live(console=console, refresh_per_second=100) if markdown else None
            if live:
                live.start()

            tool_calls_dict = {}

            # === OpenAI Streaming ===
            if self.sdk == "openai":
                async for chunk in response:
                    delta = getattr(chunk.choices[0], "delta", None)
                    finish_reason = getattr(chunk.choices[0], "finish_reason", None)
                    text = getattr(delta, "content", "") or getattr(delta, "text", "")

                    if text:
                        assistant_content += text
                        if output_callback:
                            output_callback(text)
                        elif live:
                            live.update(Markdown(assistant_content))
                        elif not markdown and not quiet:
                            print(text, end="")

                    if hasattr(delta, "tool_calls") and delta.tool_calls:
                        for tool_call_delta in delta.tool_calls:
                            index = tool_call_delta.index
                            if index not in tool_calls_dict:
                                tool_calls_dict[index] = {
                                    "id": None,
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

            # === Anthropic Streaming ===
            elif self.sdk == "anthropic":
                async for event in response:
                    if hasattr(event, "text"):
                        assistant_content += event.text
                        if output_callback:
                            output_callback(event.text)
                        elif live:
                            live.update(Markdown(assistant_content))
                        elif not markdown and not quiet:
                            print(event.text, end="")

            if live:
                live.stop()
            if not output_callback and not markdown and not quiet:
                print()

            tool_calls = list(tool_calls_dict.values())

        else:
            # Non-streaming fallback
            if self.sdk == "openai" and hasattr(response, "choices"):
                message = response.choices[0].message
                assistant_content = message.content or ""
                tool_calls = message.tool_calls if message and message.tool_calls else []

                if output_callback:
                    output_callback(assistant_content)
                elif not quiet:
                    print(assistant_content)

            elif self.sdk == "anthropic":
                parts = getattr(response, "content", [])
                assistant_content = "".join(getattr(p, "text", "") for p in parts)

                if output_callback:
                    output_callback(assistant_content)
                elif not quiet:
                    print(assistant_content)

        # Record assistant message
        assistant_msg = {
            "role": "assistant",
            "content": assistant_content
        }

        if tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["function"]["name"],
                        "arguments": call["function"]["arguments"]
                    }
                } for call in tool_calls
            ]

        self.history.append(assistant_msg)
        if self.session_enabled and self.session_id:
            self.session.msg_insert(self.session_id, assistant_msg)

        self._log(f"[ASSISTANT] {assistant_msg}")

        # Tool execution
        for call in tool_calls:
            name = call["function"]["name"]
            args = call["function"]["arguments"]
            call_id = call["id"]
            result = await self._handle_tool_call_async(name, args, call_id, kwargs, markdown, None, output_callback)

            tool_msg = {
                "role": "tool",
                "content": json.dumps(result),
                self.tool_key: call_id
            }

            self.history.append(tool_msg)
            if self.session_enabled and self.session_id:
                self.session.msg_insert(self.session_id, tool_msg)

        return assistant_content or "No response."


    async def _openai_runner(self, *, model, messages, stream):
        params = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }

        if self.tools_enabled and self.tools_supported:
            enabled_tools = self._get_enabled_tools()
            if enabled_tools:
                params["tools"] = enabled_tools
                params["tool_choice"] = "auto"

        return await self._retry_with_backoff(
            self.async_client.chat.completions.create,
            **params
        )

    async def _anthropic_runner(self, *, model, messages, stream):
        kwargs = {
            "model": model,
            "messages": messages,
            "max_tokens": 8192,
            "system": self.system
        }

        if self.tools_enabled and self.tools_supported:
            enabled_tools = [
                {
                    "name": t["function"]["name"],
                    "description": t["function"].get("description", ""),
                    "input_schema": t["function"]["parameters"]
                }
                for t in self._get_enabled_tools()
            ]
            if enabled_tools:
                kwargs["tools"] = enabled_tools

        if stream:
            return await self.async_client.messages.stream(**kwargs)
        else:
            return await self._retry_with_backoff(
                self.async_client.messages.create,
                **kwargs
            )

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
        try:
            arguments = json.loads(function_arguments or "{}")
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in tool call arguments: {e}")
            return {"error": "Invalid JSON format in tool call arguments."}

        print(f"Running: {function_name}...")

        self._log(f"[TOOL:{function_name}] args={arguments}")

        func = getattr(self, function_name, None)
        if not func:
            raise ValueError(f"Function '{function_name}' not found.")

        if live:
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

            if live:
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


    def _count_tokens(self, messages, tools=None, system=None) -> int:
        """Estimate input token count based on SDK and message structure.

        Args:
            messages (List[dict]): Message sequence in Claude or OpenAI format.
            tools (List[dict], optional): Tool specs if any.
            system (str, optional): System prompt if applicable.

        Returns:
            int: Estimated token count.
        """
        if self.sdk == "anthropic":
            try:
                response = self.client.messages.count_tokens(
                    model=self.model,
                    messages=messages,
                    tools=tools or [],
                    system=self.system
                )
                tokens_data = json.loads(response.model_dump_json())
                return token_data["input_tokens"] 
            except Exception as e:
                return self._estimate_tokens_tiktoken(messages)

        elif self.sdk == "openai":
            return self._estimate_tokens_tiktoken(messages)

        else:
            self.logger.warning(f"Token counting not supported for SDK '{self.sdk}'")
            return 0

    def _cycle_messages(self):
        """Trim messages to fit within current context window using tracked token estimates."""
        exceeded_context = False
        removed = 0

        while self.token_estimate > self.context_length and self.history:
            for i, msg in enumerate(self.history):
                if msg["role"] != "system":
                    estimated_loss = 0
                    try:
                        estimated_loss = self._count_tokens([msg])
                    except Exception:
                        pass

                    self.logger.debug(f"Trimming message to maintain context length: {msg}")
                    self.history.pop(i)
                    self.token_estimate -= estimated_loss
                    removed += 1
                    break

        if removed > 0:
            self._log(f"[TRIM] Removed {removed} messages to fit within context limit ({self.context_length})")

        if len(self.history) < 1:
            self.logger.error(f"Context length exceeded: all messages trimmed.")
            exceeded_context = True

        self._normalize_history_to_sdk()

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
        print(caller.list_models())

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
