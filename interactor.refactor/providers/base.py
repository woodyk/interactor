"""Base provider interface for LLM interactions.

This module defines the abstract base class that all LLM providers must implement.
It ensures consistent interface across different providers while allowing for
provider-specific implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable, Tuple

class BaseProvider(ABC):
    """Abstract base class for LLM providers.
    
    This class defines the interface that all LLM providers must implement.
    It ensures consistent behavior across different providers while allowing
    for provider-specific optimizations and features.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        base_url: Optional[str] = None,
        **kwargs
    ):
        """Initialize the provider.
        
        Args:
            api_key: Optional API key for the provider
            base_url: Optional base URL override
            **kwargs: Additional provider-specific options
        """
        self.api_key = api_key
        self.base_url = base_url
        self.client = None
        self.async_client = None
        self.options = kwargs
        self._setup_clients()
    
    @abstractmethod
    def _setup_clients(self):
        """Initialize provider-specific clients.
        
        This method should set up both synchronous and asynchronous clients
        for the provider's API.
        """
        pass
    
    @abstractmethod
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
        """Run a completion with the provider's API.
        
        Args:
            model: Model identifier
            messages: List of message dictionaries
            stream: Whether to stream the response
            tools: Optional list of tool definitions
            markdown: Whether to render markdown
            quiet: Whether to suppress status output
            live: Optional rich.Live instance for display
            output_callback: Optional callback function for each token
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Dict containing:
                - content: The generated text content
                - tool_calls: List of tool calls if any
        """
        pass
    
    @abstractmethod
    def normalize_messages(
        self, 
        messages: List[Dict], 
        system_message: str,
        force: bool = False
    ) -> List[Dict]:
        """Convert standardized messages to provider-specific format.
        
        Args:
            messages: List of standardized message dictionaries
            system_message: System message to include
            force: Whether to force normalization even if format hasn't changed
            
        Returns:
            List of provider-specific message dictionaries
        """
        pass
    
    @abstractmethod
    def normalize_message(self, message: Dict) -> Dict:
        """Convert a single standardized message to provider-specific format.
        
        Args:
            message: Standardized message dictionary
            
        Returns:
            Provider-specific message dictionary
        """
        pass
    
    @abstractmethod
    def supports_tools(self, model: str) -> bool:
        """Check if the current model supports tool calling.
        
        Args:
            model: Model identifier to check
        
        Returns:
            bool: True if tools are supported, False otherwise
        """
        pass
    
    @abstractmethod
    def get_token_count(self, messages: List[Dict]) -> int:
        """Get the token count for a list of messages.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            int: Estimated token count
        """
        pass
    
    @abstractmethod
    def list_models(self) -> List[str]:
        """List available models for this provider.
        
        Returns:
            List[str]: List of available model identifiers
        """
        pass
    
    @property
    def provider_name(self) -> str:
        """Get the name of this provider.
        
        Returns:
            str: Provider name
        """
        return self.__class__.__name__.replace('Provider', '').lower() 