"""Direct API client for LLM providers.

This module provides the foundation for direct API calls to LLM providers
without relying on their SDKs. This simplifies dependencies and allows for
consistent behavior across providers.
"""

import aiohttp
import json
import logging
from typing import Dict, Any, Optional, List

class DirectAPIClient:
    """Base client for direct API interactions with LLM providers.
    
    This class provides the foundation for making API requests without 
    using provider SDKs. Specific provider clients will inherit from this.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 60,
        **kwargs
    ):
        """Initialize the direct API client.
        
        Args:
            api_key: API key for the provider
            base_url: Base URL for the API
            headers: Custom headers to include in all requests
            timeout: Request timeout in seconds
            **kwargs: Additional configuration options
        """
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.session = None
        self.headers = headers or {}
        self.options = kwargs
        
        # Add API key to headers if provided
        if api_key and 'Authorization' not in self.headers:
            self.headers['Authorization'] = f'Bearer {api_key}'
            
        # Set default content type if not provided
        if 'Content-Type' not in self.headers:
            self.headers['Content-Type'] = 'application/json'
            
    async def _ensure_session(self):
        """Ensure an aiohttp session is available."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(headers=self.headers)
            
    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            
    async def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        **kwargs
    ) -> Any:
        """Make an API request.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint to call
            data: Request body data
            params: URL parameters
            stream: Whether the response should be streamed
            **kwargs: Additional request parameters
            
        Returns:
            Response data, either as JSON or streamed chunks
        """
        await self._ensure_session()
        
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        request_kwargs = {
            'params': params,
            'timeout': self.timeout,
            **kwargs
        }
        
        if data:
            request_kwargs['json'] = data
        
        try:
            async with self.session.request(method, url, **request_kwargs) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    raise Exception(f"API request failed with status {response.status}: {error_text}")
                
                if stream:
                    return response  # Return the response for streaming
                else:
                    return await response.json()
        except Exception as e:
            logging.error(f"API request error: {str(e)}")
            raise
            
    async def post(self, endpoint: str, data: Dict[str, Any], stream: bool = False, **kwargs) -> Any:
        """Make a POST request.
        
        Args:
            endpoint: API endpoint
            data: Request body
            stream: Whether to stream the response
            **kwargs: Additional request parameters
            
        Returns:
            Response data
        """
        return await self._make_request('POST', endpoint, data=data, stream=stream, **kwargs)
    
    async def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        """Make a GET request.
        
        Args:
            endpoint: API endpoint
            params: URL parameters
            **kwargs: Additional request parameters
            
        Returns:
            Response data
        """
        return await self._make_request('GET', endpoint, params=params, **kwargs)
        
    async def stream_response(self, response) -> None:
        """Process a streaming response.
        
        Args:
            response: aiohttp Response object
            
        Yields:
            Parsed chunks from the stream
        """
        async for line in response.content:
            line = line.strip()
            if not line:
                continue
                
            # Handle different streaming formats
            try:
                if line.startswith(b'data: '):
                    line = line[6:]  # Remove 'data: ' prefix
                    
                if line == b'[DONE]':
                    break
                    
                data = json.loads(line)
                yield data
            except json.JSONDecodeError:
                logging.warning(f"Failed to parse streaming response line: {line}")
                continue 