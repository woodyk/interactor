"""Grok provider implementation.

This module implements the BaseProvider interface for X.AI's Grok API.
"""

import json
import logging
import traceback
from typing import Dict, List, Optional, Any, Callable

import openai
from openai import OpenAI, AsyncOpenAI
import tiktoken

from .base import BaseProvider

class GrokProvider(BaseProvider):
    """Provider implementation for X.AI's Grok API.
    
    This class implements the BaseProvider interface for Grok.
    While Grok uses an API similar to OpenAI, there are enough differences
    to warrant a separate implementation.
    """
    
    def _setup_clients(self):
        """Initialize Grok clients.
        
        Sets up both synchronous and asynchronous clients for the Grok API.
        """
        # Use default Grok base URL if not provided
        if not self.base_url:
            self.base_url = "https://api.x.ai/v1"
            
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        self.async_client = AsyncOpenAI(
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
        **kwargs
    ) -> Dict[str, Any]:
        """Run a completion with Grok's API.
        
        This method handles Grok-specific API interactions and response processing.
        
        Args:
            model: Model identifier (e.g., "grok-1")
            messages: List of message dictionaries
            stream: Whether to stream the response
            tools: Optional list of tool definitions
            markdown: Whether to render markdown
            quiet: Whether to suppress status output
            live: Optional rich.Live instance for display
            output_callback: Optional callback function for each token
            **kwargs: Additional Grok-specific parameters
            
        Returns:
            Dict containing:
                - content: The generated text content
                - tool_calls: List of tool calls if any
        """
        logging.debug(f"[GROK REQUEST] Sending request to {model} with {len(messages)} messages")

        # Prepare API parameters
        params = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs
        }

        # Add tools if provided and supported
        if tools and self.supports_tools(model):
            params["tools"] = tools
            params["tool_choice"] = "auto"

        # Call API with retry handling
        try:
            if stream:
                response = await self.async_client.chat.completions.create(**params)
            else:
                response = await self.async_client.chat.completions.create(**params)
        except Exception:
            logging.error(f"[GROK ERROR]: {traceback.format_exc()}")
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

                # Process tool calls - Grok tool calls may have slight format differences
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    for tool_call_delta in delta.tool_calls:
                        index = getattr(tool_call_delta, "index", 0)
                        if index not in tool_calls_dict:
                            tool_calls_dict[index] = {
                                "id": getattr(tool_call_delta, "id", f"call_{index}"),
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
        """Convert standardized messages to Grok format.
        
        This method transforms the internal message format into the structure
        required by the Grok API.
        
        Args:
            messages: List of standardized message dictionaries
            system_message: System message to include
            force: Whether to force normalization even if format hasn't changed
            
        Returns:
            List of Grok-specific message dictionaries
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

            # Handle different message types - Grok mostly follows OpenAI format
            if msg["role"] == "user":
                history.append({
                    "role": "user",
                    "content": msg["content"]
                })

            elif msg["role"] == "assistant":
                # For assistant messages with tool calls
                if "metadata" in msg and msg["metadata"].get("tool_info"):
                    # This is an assistant message with tool calls
                    tool_info = msg["metadata"]["tool_info"]

                    # Create assistant message with tool calls
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
        """Convert a single standardized message to Grok format.
        
        Args:
            message: Standardized message dictionary
            
        Returns:
            Grok-specific message dictionary
        """
        role = message.get("role")
        content = message.get("content")

        if role == "system":
            return {
                "role": "system",
                "content": content
            }

        elif role == "user":
            return {
                "role": "user",
                "content": content
            }

        elif role == "assistant":
            # For assistant messages, handle potential tool calls
            if "metadata" in message and message["metadata"].get("tool_info"):
                # This is an assistant message with tool calls
                tool_info = message["metadata"]["tool_info"]
                
                # Create Grok assistant message with tool calls
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
        
        Args:
            model: Model identifier to check
            
        Returns:
            bool: True if tools are supported, False otherwise
        """
        # Known tool-supporting Grok models
        tool_supporting_models = ["grok-1", "grok-2"]
        
        for supported_model in tool_supporting_models:
            if supported_model.lower() in model.lower():
                return True

        # Try to test if the model is not in our known list but might support tools
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
            return bool(hasattr(message, "tool_calls") and message.tool_calls)
        except Exception as e:
            logging.error(f"Tool support check failed for {model}: {e}")
            return False
    
    def get_token_count(self, messages: List[Dict]) -> int:
        """Get the token count for a list of messages.
        
        Args:
            messages: List of message dictionaries
            
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
            # Base token count for message metadata
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
    
    def list_models(self) -> List[str]:
        """List available models for Grok.
        
        Returns:
            List[str]: List of available model identifiers
        """
        try:
            response = self.client.models.list()
            return [model.id for model in response.data]
        except Exception as e:
            logging.error(f"Failed to list Grok models: {e}")
            
            # Fallback: return known Grok models
            return ["grok-1", "grok-2"] 