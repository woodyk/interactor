"""Direct OpenAI API client.

This module implements direct HTTP calls to the OpenAI API without using the OpenAI SDK.
"""

import json
import logging
import uuid
from typing import Dict, List, Any, Optional, AsyncGenerator, Callable

from .direct_api_client import DirectAPIClient

class OpenAIDirectClient(DirectAPIClient):
    """Client for direct API interactions with OpenAI.
    
    This implementation makes direct HTTP calls to the OpenAI API without 
    using the OpenAI SDK.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        base_url: Optional[str] = "https://api.openai.com/v1",
        organization: Optional[str] = None,
        **kwargs
    ):
        """Initialize the OpenAI direct API client.
        
        Args:
            api_key: OpenAI API key
            base_url: Base URL for the API (default is OpenAI's API)
            organization: OpenAI organization ID
            **kwargs: Additional configuration options
        """
        headers = {}
        if organization:
            headers["OpenAI-Organization"] = organization
            
        super().__init__(api_key=api_key, base_url=base_url, headers=headers, **kwargs)
    
    async def create_chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Any:
        """Create a chat completion with the OpenAI API.
        
        Args:
            model: Model identifier (e.g., "gpt-4o")
            messages: List of message dictionaries
            stream: Whether to stream the response
            tools: Optional list of tool definitions
            tool_choice: Tool choice strategy
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            OpenAI completion response
        """
        endpoint = "chat/completions"
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs
        }
        
        # Add optional parameters if provided
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
            
        if stream:
            response = await self.post(endpoint, payload, stream=True)
            return self._process_chat_stream(response)
        else:
            return await self.post(endpoint, payload)
    
    async def _process_chat_stream(self, response) -> AsyncGenerator[Dict[str, Any], None]:
        """Process an OpenAI streaming chat completion response.
        
        Args:
            response: aiohttp Response object
            
        Yields:
            Parsed chunks from the stream
        """
        async for line in self.stream_response(response):
            if not isinstance(line, dict) or "choices" not in line:
                continue
                
            # Extract the chunk data
            chunk = {
                "id": line.get("id", ""),
                "created": line.get("created", 0),
                "model": line.get("model", ""),
                "choices": []
            }
            
            # Process choices
            for choice_data in line.get("choices", []):
                choice = {
                    "index": choice_data.get("index", 0),
                    "finish_reason": choice_data.get("finish_reason", None),
                }
                
                # Process delta
                if "delta" in choice_data:
                    delta = choice_data["delta"]
                    choice["delta"] = delta
                    
                chunk["choices"].append(choice)
                
            yield chunk
    
    async def list_models(self) -> Dict[str, Any]:
        """List available models from the OpenAI API.
        
        Returns:
            Dict containing model information
        """
        return await self.get("models")
        
    async def create_tool_result(
        self,
        thread_id: str,
        tool_call_id: str,
        output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Submit a tool result for an Assistant API tool call.
        
        Args:
            thread_id: The thread ID
            tool_call_id: The tool call ID
            output: The result of the tool execution
            
        Returns:
            API response
        """
        endpoint = f"threads/{thread_id}/runs/{tool_call_id}/submit_tool_outputs"
        
        payload = {
            "tool_outputs": [{
                "tool_call_id": tool_call_id,
                "output": json.dumps(output)
            }]
        }
        
        return await self.post(endpoint, payload)
    
    def create_tool_call(
        self,
        name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a standardized tool call object.
        
        Args:
            name: The name of the tool/function
            arguments: The arguments to pass to the tool
            
        Returns:
            Dict containing tool call information
        """
        return {
            "id": str(uuid.uuid4()),
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments)
            }
        } 