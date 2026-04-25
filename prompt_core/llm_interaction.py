"""
Main LLM interaction module with multi-provider support using instructor and litellm.
"""

import os
from typing import Optional, Literal
from dotenv import load_dotenv

from .exceptions import (
    ConfigurationError,
    APIKeyError,
    ProviderNotSupportedError,
    ProviderNotFoundError,
    InvalidResponseError,
)
from .models import EvaluationCriteria

load_dotenv()

# Try to import instructor with litellm support
try:
    import instructor
    from litellm import completion

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    raise ProviderNotFoundError(
        "litellm not installed. Install with: uv add litellm\n"
        "This package is required for multi-provider LLM support."
    )

# Provider type for type hints
ProviderType = Literal[
    "openai", "google", "anthropic", "groq", "together", "azure", "openrouter"
]


def get_client(provider: Optional[ProviderType] = None):
    """
    Get LLM client for the specified provider.

    Args:
        provider: One of "openai", "google", "anthropic", "groq", "together", "azure", "openrouter"
                 If None, uses provider from config

    Returns:
        Instructor-patched client for the specified provider

    Raises:
        ConfigurationError: If provider configuration is missing or invalid
        APIKeyError: If API key is missing for the provider
        ProviderNotSupportedError: If provider is not supported
        ProviderNotFoundError: If litellm is not installed
    """
    from .exceptions import (
        ConfigurationError,
        APIKeyError,
        ProviderNotSupportedError,
        ProviderNotFoundError,
    )

    if not LITELLM_AVAILABLE:
        raise ProviderNotFoundError(
            "litellm not installed. Install with: uv add litellm\n"
            "This package is required for multi-provider LLM support."
        )

    # Use litellm for multi-provider support
    if provider is None:
        # Get provider from config, not environment or defaults
        from .config import config

        provider = config.provider

    if not provider:
        raise ConfigurationError("LLM provider not configured in config.json")

    provider = provider.lower()

    # Map provider to litellm model name and required API key
    provider_config = {
        "openai": {
            "api_key_env": "OPENAI_API_KEY",
            "error_msg": "OPENAI_API_KEY not set. Get key from: https://platform.openai.com/api-keys",
        },
        "google": {
            "api_key_env": "GOOGLE_API_KEY",
            "error_msg": "GOOGLE_API_KEY not set. Get key from: https://makersuite.google.com/app/apikey",
        },
        "anthropic": {
            "api_key_env": "ANTHROPIC_API_KEY",
            "error_msg": "ANTHROPIC_API_KEY not set. Get key from: https://console.anthropic.com",
        },
        "groq": {
            "api_key_env": "GROQ_API_KEY",
            "error_msg": "GROQ_API_KEY not set. Get key from: https://console.groq.com",
        },
        "together": {
            "api_key_env": "TOGETHER_API_KEY",
            "error_msg": "TOGETHER_API_KEY not set. Get key from: https://together.ai",
        },
        "openrouter": {
            "api_key_env": "OPENROUTER_API_KEY",
            "error_msg": "OPENROUTER_API_KEY not set. Get key from: https://openrouter.ai/keys",
        },
    }

    if provider not in provider_config:
        raise ProviderNotSupportedError(
            f"Unsupported provider: {provider}. "
            f"Supported providers: {', '.join(provider_config.keys())}"
        )

    config = provider_config[provider]
    api_key = os.getenv(config["api_key_env"])

    if not api_key:
        raise APIKeyError(config["error_msg"])

    # Create litellm client with instructor patch
    # Use JSON mode for better compatibility with OpenRouter models
    return instructor.from_litellm(completion, mode=instructor.Mode.JSON)


def generate_evaluation_criteria(
    context: str = "birthday presents for a child",
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_retries: Optional[int] = None,
) -> EvaluationCriteria:
    """
    Generate evaluation criteria for a given context using LLM.

    Args:
        context: The context for evaluation (e.g., "birthday presents for a 7-year-old")
        model: The model to use (defaults to configured model)
        temperature: Sampling temperature (0.0 to 2.0, defaults to configured temperature)
        max_retries: Maximum number of retry attempts (defaults to configured max_retries)

    Returns:
        EvaluationCriteria object with generated criteria
    """
    from .config import config

    client = get_client()

    # Use provided values or fall back to config
    model = model or config.model
    temperature = temperature or config.temperature
    max_retries = max_retries or config.max_retries

    prompt = f"""
    You are an expert at creating evaluation criteria for decision making.
    
    Create a comprehensive set of criteria for evaluating options in this context: {context}
    
    For each criterion, provide:
    1. A clear name
    2. A detailed description of what it measures
    3. A weight from 0.0 to 10.0 indicating importance
    4. An ideal or target value if applicable
    
    The criteria should be specific, measurable, and relevant to the context.
    Include both objective and subjective criteria where appropriate.
    """

    criteria = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_retries=max_retries,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that creates structured evaluation criteria.",
            },
            {"role": "user", "content": prompt},
        ],
        response_model=EvaluationCriteria,
    )

    # Set the context on the returned object
    criteria.context = context
    return criteria


def chat_with_llm(
    user_message: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Simple chat interface with LLM.

    Args:
        user_message: The user's message
        model: The model to use (defaults to configured model)
        temperature: Sampling temperature (0.0 to 2.0, defaults to configured temperature)
        system_prompt: Optional system prompt

    Returns:
        LLM response as string
    """
    from .config import config

    client = get_client()

    # Use provided values or fall back to config
    model = model or config.model
    temperature = temperature or config.temperature

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=messages,
    )

    return response.choices[0].message.content


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
