"""Direct Anthropic API client.

This module implements direct HTTP calls to the Anthropic API without using the Anthropic SDK.
"""

import json
import logging
import uuid
from typing import Dict, List, Any, Optional, AsyncGenerator, Callable

from .direct_api_client import DirectAPIClient

class AnthropicDirectClient(DirectAPIClient):
    """Client for direct API interactions with Anthropic.
    
    This implementation makes direct HTTP calls to the Anthropic API without 
    using the Anthropic SDK.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        base_url: Optional[str] = "https://api.anthropic.com",
        version: str = "2023-06-01",
        **kwargs
    ):
        """Initialize the Anthropic direct API client.
        
        Args:
            api_key: Anthropic API key
            base_url: Base URL for the API
            version: API version
            **kwargs: Additional configuration options
        """
        headers = {
            "anthropic-version": version,
            "x-api-key": api_key  # Anthropic uses x-api-key instead of Authorization
        }
        
        # Use a different auth pattern for Anthropic
        super().__init__(api_key=None, base_url=base_url, headers=headers, **kwargs)
        
        # Store the API key separately
        self.api_key = api_key
    
    async def create_message(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Any:
        """Create a message with the Anthropic API.
        
        Args:
            model: Model identifier (e.g., "claude-3-opus-20240229")
            messages: List of message dictionaries
            system: System prompt
            stream: Whether to stream the response
            tools: Optional list of tool definitions
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            Anthropic message response
        """
        endpoint = "v1/messages"
        
        payload = {
            "model": model,
            "messages": messages,
            **kwargs
        }
        
        # Add optional parameters if provided
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = tools
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
            
        if stream:
            payload["stream"] = True
            response = await self.post(endpoint, payload, stream=True)
            return self._process_message_stream(response)
        else:
            return await self.post(endpoint, payload)
    
    async def _process_message_stream(self, response) -> AsyncGenerator[Dict[str, Any], None]:
        """Process an Anthropic streaming message response.
        
        Args:
            response: aiohttp Response object
            
        Yields:
            Parsed chunks from the stream
        """
        async for line in self.stream_response(response):
            if not isinstance(line, dict) or "type" not in line:
                continue
                
            event_type = line.get("type")
            
            # Process different event types
            if event_type == "message_start":
                # Initial message information
                yield {
                    "type": event_type,
                    "message": line.get("message", {})
                }
            elif event_type == "content_block_start":
                # Beginning of a content block (text or tool_use)
                yield {
                    "type": event_type,
                    "content_block": line.get("content_block", {})
                }
            elif event_type == "content_block_delta":
                # Incremental content update
                yield {
                    "type": event_type,
                    "delta": line.get("delta", {}),
                    "index": line.get("index", 0)
                }
            elif event_type == "content_block_stop":
                # End of a content block
                yield {
                    "type": event_type,
                    "index": line.get("index", 0)
                }
            elif event_type == "message_delta":
                # Update to message metadata
                yield {
                    "type": event_type,
                    "delta": line.get("delta", {})
                }
            elif event_type == "message_stop":
                # Final message event
                yield {
                    "type": event_type
                }
    
    async def count_tokens(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None
    ) -> Dict[str, Any]:
        """Count tokens for a set of messages using Anthropic's API.
        
        Args:
            model: Model identifier
            messages: List of message dictionaries
            system: System prompt
            
        Returns:
            Dict containing token counts
        """
        endpoint = "v1/messages/count_tokens"
        
        payload = {
            "model": model,
            "messages": messages
        }
        
        if system:
            payload["system"] = system
            
        return await self.post(endpoint, payload)
    
    def convert_openai_tools(self, openai_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI tool format to Anthropic tool format.
        
        Args:
            openai_tools: List of tools in OpenAI format
            
        Returns:
            List of tools in Anthropic format
        """
        anthropic_tools = []
        
        for tool in openai_tools:
            # Skip if not a function tool
            if tool.get("type") != "function":
                continue
                
            function = tool.get("function", {})
            
            # Extract parameters
            parameters = function.get("parameters", {})
            properties = parameters.get("properties", {})
            required = parameters.get("required", [])
            
            # Create Anthropic tool format
            anthropic_tool = {
                "name": function.get("name", ""),
                "description": function.get("description", ""),
                "input_schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
            
            anthropic_tools.append(anthropic_tool)
            
        return anthropic_tools
        
    def create_tool_use_content_block(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Create a tool use content block for an assistant message.
        
        Args:
            name: The name of the tool/function
            arguments: The arguments to pass to the tool
            
        Returns:
            Dict representing an Anthropic tool use content block
        """
        return {
            "type": "tool_use",
            "id": str(uuid.uuid4()),
            "name": name,
            "input": arguments
        }
        
    def create_tool_result_content(self, tool_use_id: str, result: Any) -> Dict[str, Any]:
        """Create a tool result content block for a user message.
        
        Args:
            tool_use_id: The ID of the tool use to respond to
            result: The result of the tool execution
            
        Returns:
            Dict representing an Anthropic tool result content
        """
        return [{
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": json.dumps(result) if isinstance(result, (dict, list)) else str(result)
        }] 