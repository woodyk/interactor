"""OpenAI provider implementation.

This module implements the BaseProvider interface for OpenAI's API.
"""

import json
import logging
import traceback
from typing import Dict, List, Optional, Any, Callable

import openai
from openai import OpenAI, AsyncOpenAI
import tiktoken

from .base import BaseProvider
from .direct_openai import OpenAIDirectClient

class OpenAIProvider(BaseProvider):
    """Provider implementation for OpenAI API.
    
    This class implements the BaseProvider interface for OpenAI's API,
    including both chat completion and tool calling features.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        base_url: Optional[str] = None,
        use_direct_api: bool = False,
        **kwargs
    ):
        """Initialize the provider.
        
        Args:
            api_key: Optional API key for the provider
            base_url: Optional base URL override
            use_direct_api: Whether to use direct API calls instead of the SDK
            **kwargs: Additional provider-specific options
        """
        self.api_key = api_key
        self.base_url = base_url
        self.client = None
        self.async_client = None
        self.direct_client = None
        self.options = kwargs
        self.use_direct_api = use_direct_api
        self._setup_clients()
    
    def _setup_clients(self):
        """Initialize OpenAI clients.
        
        Sets up both synchronous and asynchronous clients for the OpenAI API.
        If using direct API calls, also initializes the direct client.
        """
        # Set up SDK clients (always initialized for fallback)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        self.async_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # Set up direct API client if requested
        if self.use_direct_api:
            self.direct_client = OpenAIDirectClient(
                api_key=self.api_key,
                base_url=self.base_url,
                organization=self.options.get("organization")
            )
        
        # Set up encoding for token counting
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.encoding = None
            logging.warning("Failed to initialize tiktoken encoding")
    
    async def run_completion(
        self,
        *,
        model: str,
        messages: List[Dict],
        stream: bool = True,
        tools: Optional[List[Dict]] = None,
        markdown: bool = False,
        quiet: bool = False,
        live: Optional[Any] = None,
        output_callback: Optional[Callable] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Run a completion with OpenAI's API.
        
        This method handles OpenAI-specific API interactions and response processing,
        including streaming and tool calls.
        
        Args:
            model: Model identifier (e.g., "gpt-4o-mini")
            messages: List of message dictionaries in OpenAI format
            stream: Whether to stream the response
            tools: Optional list of tool definitions
            markdown: Whether to render markdown
            quiet: Whether to suppress status output
            live: Optional rich.Live instance for display
            output_callback: Optional callback function for each token
            **kwargs: Additional OpenAI-specific parameters
            
        Returns:
            Dict containing:
                - content: The generated text content
                - tool_calls: List of tool calls if any
        """
        logging.debug(f"[OPENAI REQUEST] Sending request to {model} with {len(messages)} messages")

        # Prepare API parameters - history is already normalized
        params = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs
        }

        # Add tools if provided
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"

        if self.use_direct_api:
            return await self._run_direct_completion(
                model=model, 
                messages=messages,
                stream=stream,
                tools=tools,
                markdown=markdown,
                quiet=quiet,
                live=live,
                output_callback=output_callback,
                **kwargs
            )
        else:
            return await self._run_sdk_completion(
                params=params,
                markdown=markdown,
                quiet=quiet,
                live=live,
                output_callback=output_callback
            )
    
    async def _run_sdk_completion(
        self,
        *,
        params: Dict[str, Any],
        markdown: bool = False,
        quiet: bool = False,
        live: Optional[Any] = None,
        output_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Run a completion using the OpenAI SDK.
        
        Args:
            params: Parameters for the API call
            markdown: Whether to render markdown
            quiet: Whether to suppress status output
            live: Optional rich.Live instance for display
            output_callback: Optional callback function for each token
            
        Returns:
            Dict containing completion results
        """
        stream = params.get("stream", True)
        
        # Call API
        try:
            if stream:
                response = await self.async_client.chat.completions.create(**params)
            else:
                response = await self.async_client.chat.completions.create(**params)
        except Exception:
            logging.error(f"[OPENAI ERROR RUNNER]: {traceback.format_exc()}")
            raise

        assistant_content = ""
        tool_calls_dict = {}

        # Process streaming response
        if stream and hasattr(response, "__aiter__"):
            async for chunk in response:
                delta = getattr(chunk.choices[0], "delta", None)

                # Handle content chunks
                if hasattr(delta, "content") and delta.content is not None:
                    text = delta.content
                    assistant_content += text
                    if output_callback:
                        output_callback(text)
                    elif live:
                        live.update(text)
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
            logging.debug(f"[OPENAI TOOL CALLS] Found {len(tool_calls_dict)} tool calls")
            for idx, call in tool_calls_dict.items():
                logging.debug(f"[OPENAI TOOL CALL {idx}] {call['function']['name']} with ID {call['id']}")

        # Return standardized response format
        return {
            "content": assistant_content,
            "tool_calls": list(tool_calls_dict.values())
        }
        
    async def _run_direct_completion(
        self,
        *,
        model: str,
        messages: List[Dict],
        stream: bool = True,
        tools: Optional[List[Dict]] = None,
        markdown: bool = False,
        quiet: bool = False,
        live: Optional[Any] = None,
        output_callback: Optional[Callable] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Run a completion using direct API calls.
        
        Args:
            model: Model identifier
            messages: List of message dictionaries
            stream: Whether to stream the response
            tools: Optional list of tool definitions
            markdown: Whether to render markdown
            quiet: Whether to suppress status output
            live: Optional rich.Live instance for display
            output_callback: Optional callback function for each token
            **kwargs: Additional parameters
            
        Returns:
            Dict containing completion results
        """
        if not self.direct_client:
            logging.warning("Direct API client not initialized, falling back to SDK")
            params = {
                "model": model,
                "messages": messages,
                "stream": stream,
                **kwargs
            }
            if tools:
                params["tools"] = tools
                params["tool_choice"] = "auto"
                
            return await self._run_sdk_completion(
                params=params,
                markdown=markdown,
                quiet=quiet,
                live=live,
                output_callback=output_callback
            )
        
        # Setup completion parameters
        completion_params = {
            "model": model,
            "messages": messages,
            "stream": stream
        }
        
        # Add tools if provided
        if tools:
            completion_params["tools"] = tools
            completion_params["tool_choice"] = "auto"
            
        # Add optional parameters from kwargs
        for param in ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"]:
            if param in kwargs:
                completion_params[param] = kwargs[param]
                
        assistant_content = ""
        tool_calls_dict = {}
        
        try:
            if stream:
                # Handle streaming response
                async for chunk in await self.direct_client.create_chat_completion(**completion_params):
                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta", {})
                        
                        # Process content
                        if "content" in delta and delta["content"] is not None:
                            text = delta["content"]
                            assistant_content += text
                            if output_callback:
                                output_callback(text)
                            elif live:
                                live.update(text)
                            elif not markdown and not quiet:
                                print(text, end="")
                                
                        # Process tool calls
                        if "tool_calls" in delta:
                            for tool_call in delta["tool_calls"]:
                                index = tool_call.get("index", 0)
                                
                                if index not in tool_calls_dict:
                                    tool_calls_dict[index] = {
                                        "id": tool_call.get("id", f"call_{index}"),
                                        "function": {"name": "", "arguments": ""}
                                    }
                                    
                                if "function" in tool_call:
                                    function = tool_call["function"]
                                    if "name" in function:
                                        tool_calls_dict[index]["function"]["name"] = function["name"]
                                    if "arguments" in function:
                                        tool_calls_dict[index]["function"]["arguments"] += function["arguments"]
                
                if not output_callback and not markdown and not quiet:
                    print()
            else:
                # Handle non-streaming response
                response = await self.direct_client.create_chat_completion(**completion_params)
                
                choice = response.get("choices", [{}])[0]
                message = choice.get("message", {})
                
                assistant_content = message.get("content") or ""
                
                if "tool_calls" in message:
                    for i, tool_call in enumerate(message["tool_calls"]):
                        tool_calls_dict[i] = {
                            "id": tool_call.get("id", f"call_{i}"),
                            "function": {
                                "name": tool_call.get("function", {}).get("name", ""),
                                "arguments": tool_call.get("function", {}).get("arguments", "")
                            }
                        }
                        
                if output_callback:
                    output_callback(assistant_content)
                elif not quiet:
                    print(assistant_content)
                    
        except Exception as e:
            logging.error(f"[OPENAI DIRECT API ERROR]: {traceback.format_exc()}")
            raise
            
        # Return standardized response format
        return {
            "content": assistant_content,
            "tool_calls": list(tool_calls_dict.values())
        }
    
    def normalize_messages(
        self, 
        messages: List[Dict], 
        system_message: str,
        force: bool = False
    ) -> List[Dict]:
        """Convert standardized messages to OpenAI format.
        
        This method transforms the internal message format into the structure
        required by the OpenAI API.
        
        Args:
            messages: List of standardized message dictionaries
            system_message: System message to include
            force: Whether to force normalization even if format hasn't changed
            
        Returns:
            List of OpenAI-specific message dictionaries
        """
        # Start with empty history
        history = []

        # First, add the system message at position 0
        history.append({
            "role": "system",
            "content": system_message
        })

        # Process all non-system messages
        for msg in messages:
            if msg["role"] == "system":
                continue  # Skip system messages, already handled

            # Handle different message types
            if msg["role"] == "user":
                # User messages are straightforward
                history.append({
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
                    history.append(assistant_msg)
                else:
                    # Regular assistant message
                    history.append({
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
                    history.append(tool_msg)
                    
        return history
    
    def normalize_message(self, message: Dict) -> Dict:
        """Convert a single standardized message to OpenAI format.
        
        Args:
            message: Standardized message dictionary
            
        Returns:
            OpenAI-specific message dictionary
        """
        role = message.get("role")
        content = message.get("content")

        if role == "system":
            # System messages are straightforward
            return {
                "role": "system",
                "content": content
            }

        elif role == "user":
            # User messages are straightforward
            return {
                "role": "user",
                "content": content
            }

        elif role == "assistant":
            # For assistant messages, handle potential tool calls
            if "metadata" in message and message["metadata"].get("tool_info"):
                # This is an assistant message with tool calls
                tool_info = message["metadata"]["tool_info"]
                
                # Create OpenAI assistant message with tool calls
                try:
                    arguments = tool_info.get("arguments", {})
                    arguments_str = json.dumps(arguments) if isinstance(arguments, dict) else arguments
                except:
                    arguments_str = str(arguments)
                    
                return {
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
            else:
                # Regular assistant message
                return {
                    "role": "assistant",
                    "content": content
                }
            
        elif role == "tool":
            # Tool response messages
            if "metadata" in message and "tool_info" in message["metadata"]:
                tool_info = message["metadata"]["tool_info"]
                return {
                    "role": "tool",
                    "tool_call_id": tool_info["id"],
                    "content": json.dumps(content) if isinstance(content, (dict, list)) else str(content)
                }
            
        # Default case
        return {
            "role": role,
            "content": content
        }
    
    def supports_tools(self, model: str) -> bool:
        """Check if the model supports tool calling.
        
        This method tests if the provided model supports tool/function calling.
        
        Args:
            model: Model identifier to check
            
        Returns:
            bool: True if tools are supported, False otherwise
        """
        if self.use_direct_api and self.direct_client:
            # For direct API, we rely on model name patterns instead of testing
            tool_supporting_models = [
                "gpt-4", "gpt-4o", "gpt-3.5-turbo-0125", "gpt-3.5-turbo-1106"
            ]
            
            for supported_model in tool_supporting_models:
                if supported_model in model:
                    return True
            return False
        
        # Use the SDK for testing
        try:
            response = self.client.chat.completions.create(
                model=model,
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
        except Exception as e:
            logging.error(f"Tool support check failed for {model}: {e}")
            return False
    
    def get_token_count(self, messages: List[Dict]) -> int:
        """Get the token count for a list of messages.
        
        This method provides an accurate token count for OpenAI messages,
        including tool calls.
        
        Args:
            messages: List of message dictionaries in OpenAI format
            
        Returns:
            int: Estimated token count
        """
        if not self.encoding:
            try:
                self.encoding = tiktoken.get_encoding("cl100k_base")
            except Exception:
                return sum(len(str(msg.get("content", ""))) // 4 for msg in messages)
                
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
                        
                        # Add tokens for id and type fields
                        num_tokens += len(self.encoding.encode(tool_call.get("id", "")))
                        num_tokens += len(self.encoding.encode(tool_call.get("type", "function")))
            
            # Handle tool response message format
            if msg.get("role") == "tool":
                # Add tokens for tool_call_id
                tool_id = msg.get("tool_call_id", "")
                num_tokens += len(self.encoding.encode(tool_id))
        
        # Add message end tokens
        num_tokens += 2
        
        return num_tokens
    
    async def list_models(self) -> List[str]:
        """List available models for OpenAI.
        
        Returns:
            List[str]: List of available model identifiers
        """
        if self.use_direct_api and self.direct_client:
            try:
                response = await self.direct_client.list_models()
                return [model.get("id") for model in response.get("data", [])]
            except Exception as e:
                logging.error(f"Failed to list OpenAI models via direct API: {e}")
                # Fall back to SDK
        
        try:
            response = self.client.models.list()
            return [model.id for model in response.data]
        except Exception as e:
            logging.error(f"Failed to list OpenAI models: {e}")
            
            # Fallback to common models
            return [
                "gpt-4o", 
                "gpt-4o-mini", 
                "gpt-4-turbo", 
                "gpt-4", 
                "gpt-3.5-turbo"
            ]
            
    async def close(self):
        """Close any open resources.
        
        This method ensures proper cleanup of resources, particularly
        for direct API clients.
        """
        if self.direct_client:
            await self.direct_client.close() 