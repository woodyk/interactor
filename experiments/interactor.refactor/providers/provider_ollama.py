"""Ollama provider implementation.

This module implements the BaseProvider interface for Ollama's API.
"""

import logging
from typing import List

from .provider_openai import OpenAIProvider

class OllamaProvider(OpenAIProvider):
    """Provider implementation for Ollama API.
    
    This class extends the OpenAI provider since Ollama uses an OpenAI-compatible API.
    It overrides specific behaviors that differ from the standard OpenAI implementation.
    """
    
    def _setup_clients(self):
        """Initialize Ollama clients.
        
        Sets up the clients with Ollama-specific configuration.
        """
        # Use default Ollama base URL if not provided
        if not self.base_url:
            self.base_url = "http://localhost:11434/v1"
        
        # Ollama doesn't require an API key, use a placeholder if none provided
        if not self.api_key:
            self.api_key = "ollama"
            
        # Call the parent method to set up the clients
        super()._setup_clients()
    
    def supports_tools(self, model: str) -> bool:
        """Check if the model supports tool calling.
        
        Ollama support for tools depends on the specific model.
        Some models like Llama 3 with function calling support may work.
        
        Args:
            model: Model identifier to check
            
        Returns:
            bool: True if tools are supported, False otherwise
        """
        # List of known Ollama models that support function calling
        tool_supporting_models = [
            "llama3", "llama-3", "llama3-8b", "llama3-70b", 
            "mixtral", "neural-chat"
        ]
        
        # Check if the model name contains any of the supported model identifiers
        for supported_model in tool_supporting_models:
            if supported_model.lower() in model.lower():
                logging.debug(f"[TOOLS] Ollama model {model} may support tools based on name")
                # We still try the actual check from OpenAI provider
                return super().supports_tools(model)
        
        return False
    
    def list_models(self) -> List[str]:
        """List available models for Ollama.
        
        Returns:
            List[str]: List of available model identifiers
        """
        try:
            # Ollama provides model list through a different endpoint
            response = self.client.get(f"{self.base_url}/api/tags")
            return [model["name"] for model in response.json()["models"]]
        except Exception as e:
            logging.error(f"Failed to list Ollama models: {e}")
            
            # Fallback: return a list of common Ollama models
            return [
                "llama3", "llama3-8b", "llama3-70b",
                "mistral", "mixtral", "phi",
                "neural-chat", "orca-mini", "gemma"
            ] 