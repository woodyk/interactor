"""Anthropic provider implementation.

This module implements the BaseProvider interface for Anthropic's API.
"""

import json
import logging
import traceback
from typing import Dict, List, Optional, Any, Callable

import anthropic
from anthropic import Anthropic, AsyncAnthropic
import tiktoken

from .base import BaseProvider
from .direct_anthropic import AnthropicDirectClient

class AnthropicProvider(BaseProvider):
    """Provider implementation for Anthropic API.
    
    This class implements the BaseProvider interface for Anthropic's API,
    including both chat completion and tool calling features. It supports
    both SDK-based and direct API communication approaches.
    
    The direct API approach allows for avoiding SDK dependencies, providing
    a more lightweight integration that's less vulnerable to SDK version
    compatibility issues.
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
        """Initialize Anthropic clients.
        
        Sets up both synchronous and asynchronous clients for the Anthropic API.
        If using direct API calls, also initializes the direct client.
        """
        # Set up SDK clients (always initialized for fallback)
        self.client = Anthropic(api_key=self.api_key)
        self.async_client = AsyncAnthropic(api_key=self.api_key)
        
        # Set up direct API client if requested
        if self.use_direct_api:
            self.direct_client = AnthropicDirectClient(
                api_key=self.api_key,
                base_url=self.base_url
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
        system_message: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Run a completion with Anthropic's API.
        
        This method handles Anthropic-specific API interactions and response processing,
        including streaming and tool calls.
        
        Args:
            model: Model identifier (e.g., "claude-3-sonnet-20240229")
            messages: List of message dictionaries in Anthropic format
            stream: Whether to stream the response
            tools: Optional list of tool definitions
            markdown: Whether to render markdown
            quiet: Whether to suppress status output
            live: Optional rich.Live instance for display
            output_callback: Optional callback function for each token
            system_message: System message to use
            **kwargs: Additional Anthropic-specific parameters
            
        Returns:
            Dict containing:
                - content: The generated text content
                - tool_calls: List of tool calls if any
        """
        logging.debug(f"[ANTHROPIC REQUEST] Sending request to {model} with {len(messages)} messages")
        
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
                system_message=system_message,
                **kwargs
            )
        else:
            return await self._run_sdk_completion(
                model=model,
                messages=messages,
                stream=stream,
                tools=tools,
                markdown=markdown,
                quiet=quiet,
                live=live,
                output_callback=output_callback,
                system_message=system_message,
                **kwargs
            )
    
    async def _run_sdk_completion(
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
        system_message: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Run a completion using the Anthropic SDK.
        
        Args:
            model: Model identifier
            messages: List of message dictionaries
            stream: Whether to stream the response
            tools: Optional list of tool definitions
            markdown: Whether to render markdown
            quiet: Whether to suppress status output
            live: Optional rich.Live instance for display
            output_callback: Optional callback function for each token
            system_message: System message to use
            **kwargs: Additional parameters
            
        Returns:
            Dict containing completion results
        """
        # Prepare API parameters - history is already normalized
        params = {
            "model": model,
            "messages": messages,
            "max_tokens": kwargs.pop("max_tokens", 4096),
            "system": system_message,
            **kwargs
        }
        
        # Add tools support if needed
        if tools:
            anthropic_tools = []
            for tool in tools:
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
                    
                anthropic_tools.append(format_tool)
            
            params["tools"] = anthropic_tools
        
        assistant_content = ""
        tool_calls_dict = {}
        
        try:
            # Process streaming response
            if stream:
                stream_params = params.copy()
                stream_params["stream"] = True
                
                stream_response = await self.async_client.messages.create(**stream_params)
               
                content_type = None
                async for chunk in stream_response:
                    chunk_type = getattr(chunk, "type", "unknown")
                    logging.debug(f"[ANTHROPIC CHUNK] Type: {chunk_type}")
                    
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
                            logging.debug(f"[ANTHROPIC TOOL USE] {tool_name}")
                    
                    # Handle text content
                    if chunk_type == "content_block_delta" and hasattr(chunk.delta, "text"):
                        delta = chunk.delta.text
                        assistant_content += delta
                        if output_callback:
                            output_callback(delta)
                        elif live:
                            live.update(delta)
                        elif not markdown and not quiet:
                            print(delta, end="")

                    # Handle complete tool use
                    elif chunk_type == "content_block_delta" and content_type == "tool_use":
                        if tool_id in tool_calls_dict and hasattr(chunk.delta, "partial_json"):
                            tool_calls_dict[tool_id]["function"]["arguments"] += chunk.delta.partial_json

            # Process non-streaming response
            else:
                # For non-streaming, ensure we don't send the stream parameter
                non_stream_params = params.copy()
                non_stream_params.pop("stream", None)  # Remove stream if it exists
                
                response = await self.async_client.messages.create(**non_stream_params)

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
                        logging.debug(f"[ANTHROPIC TOOL USE] {tool_name}")
                
                if output_callback:
                    output_callback(assistant_content)
                elif not quiet:
                    print(assistant_content)
                    
        except Exception as e:
            logging.error(f"[ANTHROPIC ERROR RUNNER] {traceback.format_exc()}")
            
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
        system_message: str = None,
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
            system_message: System message to use
            **kwargs: Additional parameters
            
        Returns:
            Dict containing completion results
        """
        max_tokens = kwargs.pop("max_tokens", 4096)
        temperature = kwargs.pop("temperature", None)
        
        anthropic_tools = None
        if tools:
            anthropic_tools = self.direct_client.convert_openai_tools(tools)
            
        assistant_content = ""
        tool_calls_dict = {}
        
        try:
            # Call the direct API client
            response_stream = await self.direct_client.create_message(
                model=model,
                messages=messages,
                system=system_message,
                stream=stream,
                tools=anthropic_tools,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            if stream:
                content_type = None
                current_tool_id = None
                
                async for chunk in response_stream:
                    chunk_type = chunk.get("type", "unknown")
                    
                    # Content block start - initialize content type tracking
                    if chunk_type == "content_block_start":
                        content_block = chunk.get("content_block", {})
                        content_type = content_block.get("type")
                        
                        # Initialize tool use tracking
                        if content_type == "tool_use":
                            current_tool_id = content_block.get("id")
                            tool_name = content_block.get("name")
                            tool_calls_dict[current_tool_id] = {
                                "id": current_tool_id,
                                "function": {
                                    "name": tool_name,
                                    "arguments": ""
                                }
                            }
                    
                    # Process content deltas
                    elif chunk_type == "content_block_delta":
                        delta = chunk.get("delta", {})
                        
                        # Handle text content
                        if "text" in delta and content_type == "text":
                            text = delta["text"]
                            assistant_content += text
                            if output_callback:
                                output_callback(text)
                            elif live:
                                live.update(text)
                            elif not markdown and not quiet:
                                print(text, end="")
                        
                        # Handle tool use content
                        elif "partial_json" in delta and content_type == "tool_use" and current_tool_id:
                            partial_json = delta["partial_json"]
                            tool_calls_dict[current_tool_id]["function"]["arguments"] += partial_json
            else:
                # Handle non-streaming response
                content_blocks = response_stream.get("content", [])
                
                for block in content_blocks:
                    block_type = block.get("type")
                    
                    if block_type == "text":
                        assistant_content += block.get("text", "")
                    
                    elif block_type == "tool_use":
                        tool_id = block.get("id")
                        tool_name = block.get("name")
                        tool_input = block.get("input", {})
                        
                        if isinstance(tool_input, dict):
                            tool_input_str = json.dumps(tool_input)
                        else:
                            tool_input_str = str(tool_input)
                            
                        tool_calls_dict[tool_id] = {
                            "id": tool_id,
                            "function": {
                                "name": tool_name,
                                "arguments": tool_input_str
                            }
                        }
                
                if output_callback:
                    output_callback(assistant_content)
                elif not quiet:
                    print(assistant_content)
        
        except Exception as e:
            logging.error(f"[ANTHROPIC DIRECT API ERROR] {traceback.format_exc()}")
            
            return {
                "content": f"Error with direct API call to Anthropic: {str(e)}",
                "tool_calls": []
            }
        
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
        """Convert standardized messages to Anthropic format.
        
        This method transforms the internal message format into the structure
        required by the Anthropic API. Note that Anthropic handles the system message
        separately from the history.
        
        Args:
            messages: List of standardized message dictionaries
            system_message: System message to include (stored separately)
            force: Whether to force normalization even if format hasn't changed
            
        Returns:
            List of Anthropic-specific message dictionaries
        """
        # Start with empty history
        history = []

        # Process all non-system messages
        for msg in messages:
            if msg["role"] == "system":
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
                    history.append(tool_result_msg)
                else:
                    # Regular user message
                    history.append({
                        "role": "user",
                        "content": msg["content"]
                    })
            
            elif msg["role"] == "assistant":
                # For assistant messages, check for tool use
                if "metadata" in msg and "tool_info" in msg["metadata"]:
                    # This is an assistant message with tool use
                    tool_info = msg["metadata"]["tool_info"]

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
                    history.append({
                        "role": "assistant",
                        "content": content_blocks
                    })
                else:
                    # Regular assistant message
                    history.append({
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
                    history.append(tool_result_msg)
                    
        return history
    
    def normalize_message(self, message: Dict) -> Dict:
        """Convert a single standardized message to Anthropic format.
        
        Args:
            message: Standardized message dictionary
            
        Returns:
            Anthropic-specific message dictionary
        """
        role = message.get("role")
        content = message.get("content")

        if role == "system":
            # System message is handled separately in Anthropic
            return None

        elif role == "user":
            # User messages - check if it contains tool results
            if "metadata" in message and "tool_info" in message["metadata"]:
                tool_info = message["metadata"]["tool_info"]
                # Check for result or directly use content
                result_content = tool_info.get("result", content)

                # Create Anthropic tool result format
                try:
                    result_str = json.dumps(result_content) if isinstance(result_content, (dict, list)) else str(result_content)
                except:
                    result_str = str(result_content)

                return {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_info["id"],
                        "content": result_str
                    }]
                }
            else:
                # Regular user message
                return {
                    "role": "user",
                    "content": content
                }

        elif role == "assistant":
            # For assistant messages, check for tool use
            if "metadata" in message and "tool_info" in message["metadata"]:
                # This is an assistant message with tool use
                tool_info = message["metadata"]["tool_info"]

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
                return {
                    "role": "assistant",
                    "content": content_blocks
                }
            else:
                # Regular assistant message
                return {
                    "role": "assistant",
                    "content": content
                }

        elif role == "tool":
            # Tool messages in standard format get converted to user messages with tool_result
            if "metadata" in message and "tool_info" in message["metadata"]:
                tool_info = message["metadata"]["tool_info"]

                try:
                    result_str = json.dumps(content) if isinstance(content, (dict, list)) else str(content)
                except:
                    result_str = str(content)

                return {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_info["id"],
                        "content": result_str
                    }]
                }
        
        # Default case for unsupported role types
        return {
            "role": "user" if role == "system" else role,
            "content": content
        }
    
    def supports_tools(self, model: str) -> bool:
        """Check if the model supports tool calling.
        
        For Claude models, we pre-define support based on model ID as well
        as try a simple test call.
        
        Args:
            model: Model identifier to check
            
        Returns:
            bool: True if tools are supported, False otherwise
        """
        # Known tool-supporting Claude models
        claude_tool_models = ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku",
                              "claude-3.5-sonnet", "claude-3.7-sonnet"]

        # Check if the current model supports tools based on name pattern
        for supported_model in claude_tool_models:
            if supported_model in model.lower():
                logging.debug(f"[TOOLS] Anthropic model {model} is known to support tools")
                return True
                
        # If using direct API and not explicitly supported by name, don't attempt test calls
        # as they could be expensive and the name check is reliable for Anthropic models
        if self.use_direct_api:
            # Could add a direct API check here if needed in the future
            return False

        # If not explicitly supported and using SDK, try to test with the SDK
        try:
            _ = self.client.messages.create(
                model=model,
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
                logging.debug(f"[TOOLS] Anthropic model {model} does not support tools: {e}")
                return False
            if "not a supported tool field" in error_msg:
                logging.debug(f"[TOOLS] Anthropic API rejected tool format: {e}")
                return False
            raise
        except Exception as e:
            logging.error(f"[TOOLS] Unexpected error testing tool support: {e}")
            return False
    
    def get_token_count(self, messages: List[Dict]) -> int:
        """Get the token count for a list of messages.
        
        For Anthropic, we attempt to use their built-in token counter first,
        then fall back to a tiktoken estimate if that fails.
        
        Args:
            messages: List of message dictionaries in Anthropic format
            
        Returns:
            int: Estimated token count
        """
        # Extract system message if present
        system = next((msg["content"] for msg in messages if msg.get("role") == "system"), None)
        non_system_messages = [msg for msg in messages if msg.get("role") != "system"]
        
        # Try using the direct API client for token counting if enabled
        if self.use_direct_api and self.direct_client:
            try:
                # Use direct API client for token counting
                token_count_future = self.direct_client.count_tokens(
                    model="claude-3-sonnet-20240229",  # Use a standard model for counting
                    messages=non_system_messages,
                    system=system
                )
                
                # Run the async operation in an event loop
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Create a new loop for the blocking call if needed
                        token_count = asyncio.run_coroutine_threadsafe(token_count_future, loop).result()
                    else:
                        token_count = loop.run_until_complete(token_count_future)
                    
                    return token_count.get("input_tokens", 0)
                except RuntimeError:
                    # If there's no running event loop
                    token_count = asyncio.run(token_count_future)
                    return token_count.get("input_tokens", 0)
            except Exception as e:
                logging.debug(f"[TOKEN COUNT] Error using direct API token counter: {e}")
                # Fall back to SDK or tiktoken estimation
        
        # Try to use Claude's built-in token counter via SDK
        try:
            if non_system_messages and self.client:
                response = self.client.messages.count_tokens(
                    model="claude-3-sonnet-20240229",  # Use a standard model for counting
                    messages=non_system_messages,
                    system=system
                )
                return response.input_tokens
        except Exception as e:
            logging.debug(f"[TOKEN COUNT] Error using Anthropic SDK token counter: {e}")
        
        # Fall back to tiktoken estimation
        if not self.encoding:
            try:
                self.encoding = tiktoken.get_encoding("cl100k_base")
            except Exception:
                return sum(len(str(msg.get("content", ""))) // 4 for msg in messages)
                
        num_tokens = 0
        
        # Count tokens for each message
        for msg in messages:
            # Base token count for message metadata
            num_tokens += 4  # Message overhead
            
            # Add tokens for role name
            role = msg.get("role", "")
            num_tokens += len(self.encoding.encode(role))
            
            # Count tokens in message content
            if isinstance(msg.get("content"), str):
                content = msg.get("content", "")
                num_tokens += len(self.encoding.encode(content))
                
            elif isinstance(msg.get("content"), list):
                # Handle Anthropic-style content lists
                for item in msg.get("content", []):
                    if isinstance(item, dict):
                        if item.get("type") == "tool_result":
                            result_content = item.get("content", "")
                            if isinstance(result_content, str):
                                num_tokens += len(self.encoding.encode(result_content))
                            else:
                                num_tokens += len(self.encoding.encode(json.dumps(result_content)))
                            # Add tokens for tool_use_id and type fields
                            num_tokens += len(self.encoding.encode(item.get("type", "")))
                            num_tokens += len(self.encoding.encode(item.get("tool_use_id", "")))
                        
                        elif item.get("type") == "text":
                            num_tokens += len(self.encoding.encode(item.get("text", "")))
                        
                        elif item.get("type") == "tool_use":
                            num_tokens += len(self.encoding.encode(item.get("name", "")))
                            tool_input = item.get("input", {})
                            if isinstance(tool_input, str):
                                num_tokens += len(self.encoding.encode(tool_input))
                            else:
                                num_tokens += len(self.encoding.encode(json.dumps(tool_input)))
                            num_tokens += len(self.encoding.encode(item.get("id", "")))
        
        # Add message end tokens
        num_tokens += 2
        
        return num_tokens
    
    def list_models(self) -> List[str]:
        """List available models for Anthropic.
        
        Returns:
            List[str]: List of available model identifiers
        """
        # Try direct API client if available
        if self.use_direct_api and self.direct_client:
            try:
                # Note: Anthropic doesn't have a formal models endpoint yet
                # This is a placeholder for when they add one
                # For now we just fall back to the static list
                pass
            except Exception as e:
                logging.error(f"Failed to list Anthropic models via direct API: {e}")
                # Fall back to static list
        
        # Anthropic API does not have a models endpoint
        # Return a list of known Claude models
        return [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0",
            "claude-instant-1.2"
        ]
        
    async def close(self):
        """Close any open resources.
        
        This method ensures proper cleanup of resources, particularly
        for direct API clients.
        """
        if self.direct_client:
            await self.direct_client.close() 