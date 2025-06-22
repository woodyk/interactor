"""Nvidia provider implementation.

This module implements the BaseProvider interface for Nvidia's API.
"""

import logging
from typing import List

from .provider_openai import OpenAIProvider

class NvidiaProvider(OpenAIProvider):
    """Provider implementation for Nvidia AI Foundation Models API.
    
    This class extends the OpenAI provider since Nvidia's API is OpenAI-compatible.
    It overrides specific behaviors that differ from the standard OpenAI implementation.
    """
    
    def _setup_clients(self):
        """Initialize Nvidia clients.
        
        Sets up the clients with Nvidia-specific configuration.
        """
        # Use default Nvidia base URL if not provided
        if not self.base_url:
            self.base_url = "https://integrate.api.nvidia.com/v1"
        
        # Call the parent method to set up the clients
        super()._setup_clients()
    
    def supports_tools(self, model: str) -> bool:
        """Check if the model supports tool calling.
        
        Nvidia API tools support depends on the specific model.
        
        Args:
            model: Model identifier to check
            
        Returns:
            bool: True if tools are supported, False otherwise
        """
        # Known tool-supporting models
        tool_supporting_models = [
            "mixtral", "llama-3", "claude"
        ]
        
        # Check if the model name contains any of the supported model identifiers
        for supported_model in tool_supporting_models:
            if supported_model.lower() in model.lower():
                logging.debug(f"[TOOLS] Nvidia model {model} may support tools based on name")
                return super().supports_tools(model)
        
        return False
    
    def list_models(self) -> List[str]:
        """List available models for Nvidia.
        
        Returns:
            List[str]: List of available model identifiers
        """
        # Try the standard OpenAI models endpoint first
        try:
            return super().list_models()
        except Exception as e:
            logging.error(f"Failed to list Nvidia models: {e}")
            
            # Fallback: return a list of known Nvidia models
            return [
                "ai-mixtral-8x7b-instruct",
                "ai-llama3-70b",
                "ai-claude-3-opus-20240229",
                "ai-claude-3-sonnet-20240229"
            ] 