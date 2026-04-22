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
    print("Warning: litellm not installed. Install with: pip install litellm")
    print("Falling back to OpenAI-only mode.")

# Provider type for type hints
ProviderType = Literal["openai", "google", "anthropic", "groq", "together", "azure"]


def get_client(provider: Optional[ProviderType] = None):
    """
    Get LLM client for the specified provider.
    
    Args:
        provider: One of "openai", "google", "anthropic", "groq", "together", "azure"
                 If None, uses LLM_PROVIDER from env or defaults to "openai"
                 
    Returns:
        Instructor-patched client for the specified provider
        
    Raises:
        ValueError: If provider not supported or API key missing
        ImportError: If litellm not installed for multi-provider support
    """
    if not LITELLM_AVAILABLE:
        # Fall back to original OpenAI-only implementation
        from openai import OpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. "
                "Please set it in a .env file or export it."
            )
        
        client = OpenAI(api_key=api_key)
        return instructor.from_openai(client)
    
    # Use litellm for multi-provider support
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "openai").lower()
    
    # Map provider to litellm model name and required API key
    provider_config = {
        "openai": {
            "model": "gpt-4o",
            "api_key_env": "OPENAI_API_KEY",
            "error_msg": "OPENAI_API_KEY not set. Get key from: https://platform.openai.com/api-keys"
        },
        "google": {
            "model": "gemini/gemini-1.5-flash",
            "api_key_env": "GOOGLE_API_KEY", 
            "error_msg": "GOOGLE_API_KEY not set. Get key from: https://makersuite.google.com/app/apikey"
        },
        "anthropic": {
            "model": "claude-3-haiku-20240307",
            "api_key_env": "ANTHROPIC_API_KEY",
            "error_msg": "ANTHROPIC_API_KEY not set. Get key from: https://console.anthropic.com"
        },
        "groq": {
            "model": "groq/llama3-70b-8192",
            "api_key_env": "GROQ_API_KEY",
            "error_msg": "GROQ_API_KEY not set. Get key from: https://console.groq.com"
        },
        "together": {
            "model": "together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "api_key_env": "TOGETHER_API_KEY",
            "error_msg": "TOGETHER_API_KEY not set. Get key from: https://together.ai"
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
    return instructor.from_litellm(completion)


def get_model_for_provider(provider: Optional[ProviderType] = None) -> str:
    """
    Get default model for the specified provider.
    
    Args:
        provider: LLM provider
        
    Returns:
        Default model name for the provider
    """
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "openai").lower()
    
    model_map = {
        "openai": "gpt-4o-mini",  # Cheaper for testing
        "google": "gemini/gemini-1.5-flash",
        "anthropic": "claude-3-haiku-20240307",
        "groq": "groq/llama3-70b-8192",
        "together": "together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "azure": "azure/gpt-4"  # Requires additional configuration
    }
    
    return model_map.get(provider, "gpt-4o-mini")


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
    }
    
    return providers


def get_default_provider() -> str:
    """
    Get the default provider based on available API keys.
    
    Returns:
        Name of the first available provider, or "openai" as fallback
    """
    available = list_available_providers()
    
    # Check for Google first (best free option)
    if available.get("google"):
        return "google"
    
    # Check for Groq (good free option)
    if available.get("groq"):
        return "groq"
    
    # Check for Together AI (has free credits)
    if available.get("together"):
        return "together"
    
    # Check for Anthropic (has $5 credit)
    if available.get("anthropic"):
        return "anthropic"
    
    # Fall back to OpenAI (has $5 credit)
    if available.get("openai"):
        return "openai"
    
    # No providers available
    return "openai"


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