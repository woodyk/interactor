"""Provider interfaces for LLM interactions.

This package contains provider-specific implementations for different LLM services.
Each provider implements a common interface defined in the base.py module.
"""

from typing import Dict, Type, Optional

from .base import BaseProvider
from .provider_openai import OpenAIProvider
from .provider_anthropic import AnthropicProvider
from .provider_ollama import OllamaProvider
from .provider_nvidia import NvidiaProvider
from .provider_google import GoogleProvider
from .provider_mistral import MistralProvider
from .provider_deepseek import DeepseekProvider
from .provider_grok import GrokProvider

# Register all available providers
PROVIDERS = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "ollama": OllamaProvider,
    "nvidia": NvidiaProvider,
    "google": GoogleProvider,
    "mistral": MistralProvider,
    "deepseek": DeepseekProvider,
    "grok": GrokProvider
}

def get_provider(name: str, **kwargs) -> BaseProvider:
    """Get a provider instance by name.
    
    Args:
        name: Provider name
        **kwargs: Provider configuration options
        
    Returns:
        An instance of the requested provider
        
    Raises:
        ValueError: If the provider is not supported
    """
    if name not in PROVIDERS:
        raise ValueError(f"Unsupported provider: {name}. Supported: {list(PROVIDERS.keys())}")
    
    return PROVIDERS[name](**kwargs) 