"""
Abstract LLM provider interface.
Keeps core logic completely provider-agnostic.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def create_structured_response(
        self,
        model: str,
        messages: List[Dict[str, str]],
        response_model: Type[T],
        max_retries: int = 3,
        **kwargs
    ) -> T:
        """
        Generate a structured response from the LLM.
        
        Args:
            model: Model identifier
            messages: Conversation history
            response_model: Pydantic model for structured response
            max_retries: Maximum retry attempts for validation
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Structured response as instance of response_model
            
        Raises:
            ValueError: If validation fails after max_retries
            ConnectionError: If API call fails
            AuthenticationError: If API key is invalid
        """
        pass


class LLMCallError(Exception):
    """Base exception for LLM call failures."""
    pass


class ValidationError(LLMCallError):
    """Raised when response validation fails."""
    pass


class AuthenticationError(LLMCallError):
    """Raised when authentication fails."""
    pass


class ConnectionError(LLMCallError):
    """Raised when connection fails."""
    pass


def get_provider(provider_name: Optional[str] = None, **kwargs) -> LLMProvider:
    """
    Factory function to get LLM provider.
    
    Args:
        provider_name: Name of provider ("openai", "google", "anthropic", etc.)
                      If None, uses LLM_PROVIDER env var or defaults to available provider
        **kwargs: Provider-specific configuration
        
    Returns:
        LLMProvider instance
        
    Raises:
        ValueError: If provider not found or not configured
    """
    import os
    from typing import Dict, Type
    
    # Map provider names to their implementations
    provider_registry: Dict[str, Type[LLMProvider]] = {}
    
    # Try to import providers dynamically
    # OpenAI provider
    try:
        from .llm_providers.openai_provider import OpenAIProvider
        provider_registry["openai"] = OpenAIProvider
    except ImportError:
        pass
    
    # Note: We could add more providers here as they're implemented
    # e.g., GoogleProvider, AnthropicProvider, GroqProvider, etc.
    
    # Use environment variable if not specified
    if provider_name is None:
        provider_name = os.getenv("LLM_PROVIDER", "openai")  # Default to openai
    
    # Auto-detect based on API keys if provider not found
    if provider_name not in provider_registry:
        # Check which providers have API keys available
        available_keys = {
            "openai": os.getenv("OPENAI_API_KEY"),
            # Add other providers here as they're implemented
        }
        
        # Find first available provider
        for provider, api_key in available_keys.items():
            if api_key and provider in provider_registry:
                provider_name = provider
                break
        else:
            # No providers with API keys found
            raise ValueError(
                f"Provider '{provider_name}' not available and no other providers configured.\n"
                f"Available providers: {', '.join(sorted(provider_registry.keys()))}\n"
                "Set required API keys in environment or install provider packages."
            )
    
    if provider_name not in provider_registry:
        available = ", ".join(sorted(provider_registry.keys()))
        raise ValueError(
            f"Provider '{provider_name}' not available. "
            f"Available providers: {available}"
        )
    
    return provider_registry[provider_name](**kwargs)