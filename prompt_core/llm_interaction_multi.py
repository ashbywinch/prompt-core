"""
Multi-provider LLM interaction using instructor with litellm.
Supports OpenAI, Google Gemini, Anthropic Claude, Groq, Together AI, and more.
"""
import os
from typing import Optional, Literal
from dotenv import load_dotenv

load_dotenv()

# Try to import instructor with litellm support
try:
    import instructor
    from litellm import completion
    from litellm.exceptions import AuthenticationError
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    raise ImportError(
        "litellm not installed. Install with: uv add litellm\n"
        "This package is required for multi-provider LLM support."
    )

# Provider type for type hints
ProviderType = Literal["openai", "google", "anthropic", "groq", "together", "azure", "openrouter"]


def get_client(provider: Optional[ProviderType] = None):
    """
    Get LLM client for the specified provider.
    
    Args:
        provider: One of "openai", "google", "anthropic", "groq", "together", "azure", "openrouter"
                 If None, uses provider from config
                 
    Returns:
        Instructor-patched client for the specified provider
        
    Raises:
        ValueError: If provider not supported or API key missing
        ImportError: If litellm not installed for multi-provider support
    """
    if not LITELLM_AVAILABLE:
        raise ImportError(
            "litellm not installed. Install with: uv add litellm\n"
            "This package is required for multi-provider LLM support."
        )
    
    # Use litellm for multi-provider support
    if provider is None:
        # Get provider from config, not environment or defaults
        from .config import config
        provider = config.provider
    
    provider = provider.lower()
    
    # Map provider to litellm model name and required API key
    provider_config = {
        "openai": {
            "api_key_env": "OPENAI_API_KEY",
            "error_msg": "OPENAI_API_KEY not set. Get key from: https://platform.openai.com/api-keys"
        },
        "google": {
            "api_key_env": "GOOGLE_API_KEY", 
            "error_msg": "GOOGLE_API_KEY not set. Get key from: https://makersuite.google.com/app/apikey"
        },
        "anthropic": {
            "api_key_env": "ANTHROPIC_API_KEY",
            "error_msg": "ANTHROPIC_API_KEY not set. Get key from: https://console.anthropic.com"
        },
        "groq": {
            "api_key_env": "GROQ_API_KEY",
            "error_msg": "GROQ_API_KEY not set. Get key from: https://console.groq.com"
        },
        "together": {
            "api_key_env": "TOGETHER_API_KEY",
            "error_msg": "TOGETHER_API_KEY not set. Get key from: https://together.ai"
        },
        "openrouter": {
            "api_key_env": "OPENROUTER_API_KEY",
            "error_msg": "OPENROUTER_API_KEY not set. Get key from: https://openrouter.ai/keys"
        }
    }
    
    if provider not in provider_config:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Supported providers: {', '.join(provider_config.keys())}"
        )
    
    config = provider_config[provider]
    api_key = os.getenv(config["api_key_env"])
    
    if not api_key:
        raise ValueError(f"{config['error_msg']}")
    
    # Create litellm client with instructor patch
    # Use JSON mode for better compatibility with OpenRouter models
    return instructor.from_litellm(completion, mode=instructor.Mode.JSON)


def get_model_for_provider(provider: Optional[ProviderType] = None) -> str:
    """
    Get default model for the specified provider.
    
    Args:
        provider: LLM provider
        
    Returns:
        Default model name for the provider
    """
    from .config import config
    
    if provider is None:
        provider = config.provider
    
    # Get the default model for the requested provider from config
    defaults = config.get("llm.defaults", {})
    return defaults.get(provider, defaults.get("openai", "gpt-4o-mini"))


def list_available_providers() -> dict:
    """
    List available LLM providers based on API keys in environment.
    
    Returns:
        Dictionary mapping provider names to availability status
    """
    providers = {
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "google": bool(os.getenv("GOOGLE_API_KEY")),
        "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
        "groq": bool(os.getenv("GROQ_API_KEY")),
        "together": bool(os.getenv("TOGETHER_API_KEY")),
        "openrouter": bool(os.getenv("OPENROUTER_API_KEY")),
    }
    
    return providers





# Legacy function for backward compatibility
def get_client_legacy():
    """
    Legacy function for backward compatibility.
    Uses the original OpenAI-only implementation.
    """
    from openai import OpenAI
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. "
            "Please set it in a .env file or export it."
        )
    
    client = OpenAI(api_key=api_key)
    
    # Try to patch with instructor
    try:
        import instructor
        return instructor.from_openai(client)
    except ImportError:
        # If instructor not available, return plain client
        return client