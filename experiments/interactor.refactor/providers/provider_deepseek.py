"""Deepseek provider implementation.

This module implements the BaseProvider interface for Deepseek's API.
"""

import logging
from typing import List

from .provider_openai import OpenAIProvider

class DeepseekProvider(OpenAIProvider):
    """Provider implementation for Deepseek AI API.
    
    This class extends the OpenAI provider since Deepseek uses an OpenAI-compatible API.
    It overrides specific behaviors that differ from the standard OpenAI implementation.
    """
    
    def _setup_clients(self):
        """Initialize Deepseek clients.
        
        Sets up the clients with Deepseek-specific configuration.
        """
        # Use default Deepseek base URL if not provided
        if not self.base_url:
            self.base_url = "https://api.deepseek.com"
        
        # Call the parent method to set up the clients
        super()._setup_clients()
    
    def supports_tools(self, model: str) -> bool:
        """Check if the model supports tool calling.
        
        Deepseek's API tools support depends on the specific model.
        
        Args:
            model: Model identifier to check
            
        Returns:
            bool: True if tools are supported, False otherwise
        """
        # Known tool-supporting Deepseek models
        tool_supporting_models = [
            "deepseek-coder", "deepseek-chat"
        ]
        
        # Check if the model name contains any of the supported model identifiers
        for supported_model in tool_supporting_models:
            if supported_model.lower() in model.lower():
                logging.debug(f"[TOOLS] Deepseek model {model} may support tools based on name")
                return super().supports_tools(model)
        
        return False
    
    def list_models(self) -> List[str]:
        """List available models for Deepseek.
        
        Returns:
            List[str]: List of available model identifiers
        """
        # Try the standard OpenAI models endpoint first
        try:
            return super().list_models()
        except Exception as e:
            logging.error(f"Failed to list Deepseek models: {e}")
            
            # Fallback: return a list of known Deepseek models
            return [
                "deepseek-chat",
                "deepseek-coder",
                "deepseek-llm-67b",
                "deepseek-coder-instruct",
                "deepseek-coder-v2"
            ] 