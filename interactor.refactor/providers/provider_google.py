"""Google provider implementation.

This module implements the BaseProvider interface for Google's AI models.
"""

import logging
from typing import List, Dict, Any, Optional, Callable

from .provider_openai import OpenAIProvider

class GoogleProvider(OpenAIProvider):
    """Provider implementation for Google's AI models API.
    
    This class extends the OpenAI provider since Google offers an OpenAI-compatible API.
    It overrides specific behaviors that differ from the standard OpenAI implementation.
    """
    
    def _setup_clients(self):
        """Initialize Google clients.
        
        Sets up the clients with Google-specific configuration.
        """
        # Use default Google base URL if not provided
        if not self.base_url:
            self.base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
        
        # Call the parent method to set up the clients
        super()._setup_clients()
    
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
        """Run a completion with Google's API.
        
        Adds Google-specific parameters to the request.
        
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
            Dict containing:
                - content: The generated text content
                - tool_calls: List of tool calls if any
        """
        # Add Google-specific parameters
        google_kwargs = {
            "safety_settings": self.options.get("safety_settings", []),
            **kwargs
        }
        
        # Call the parent method
        return await super().run_completion(
            model=model,
            messages=messages,
            stream=stream,
            tools=tools,
            markdown=markdown,
            quiet=quiet,
            live=live,
            output_callback=output_callback,
            **google_kwargs
        )
    
    def supports_tools(self, model: str) -> bool:
        """Check if the model supports tool calling.
        
        Google API tools support depends on the specific model.
        
        Args:
            model: Model identifier to check
            
        Returns:
            bool: True if tools are supported, False otherwise
        """
        # Known tool-supporting Google models (Gemini models)
        tool_supporting_models = [
            "gemini-pro", "gemini-1.5", "gemini-1.5-pro"
        ]
        
        # Check if the model name contains any of the supported model identifiers
        for supported_model in tool_supporting_models:
            if supported_model.lower() in model.lower():
                logging.debug(f"[TOOLS] Google model {model} may support tools based on name")
                return super().supports_tools(model)
        
        return False
    
    def list_models(self) -> List[str]:
        """List available models for Google.
        
        Returns:
            List[str]: List of available model identifiers
        """
        # Try the standard OpenAI models endpoint first
        try:
            return super().list_models()
        except Exception as e:
            logging.error(f"Failed to list Google models: {e}")
            
            # Fallback: return a list of known Google Gemini models
            return [
                "gemini-1.5-pro",
                "gemini-1.5-flash",
                "gemini-1.0-pro",
                "gemini-1.0-pro-vision"
            ] 