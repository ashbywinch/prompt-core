import os
import instructor
from openai import OpenAI
from typing import Optional
from dotenv import load_dotenv
from .models import EvaluationCriteria
from .conversation import ConversationOrchestrator, ConversationResult

load_dotenv()


def get_client():
    """
    Get LLM client for the specified provider.
    
    Uses multi-provider support if litellm is installed,
    otherwise falls back to OpenAI-only mode.
    
    Returns:
        Instructor-patched client for the LLM provider
        
    Raises:
        ValueError: If no API key is available for any provider
    """
    # Try to use multi-provider version first
    try:
        from .llm_interaction_multi import get_client as get_multi_client
        return get_multi_client()
    except ImportError:
        # Fall back to original OpenAI-only implementation
        pass
    
    # Original OpenAI-only implementation
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Check if other API keys are available
        other_keys = [
            ("GOOGLE_API_KEY", "Google Gemini"),
            ("GROQ_API_KEY", "Groq"),
            ("ANTHROPIC_API_KEY", "Anthropic Claude"),
            ("TOGETHER_API_KEY", "Together AI"),
        ]
        
        available = []
        for key_name, provider_name in other_keys:
            if os.getenv(key_name):
                available.append(f"{provider_name} (set {key_name})")
        
        if available:
            raise ValueError(
                f"OPENAI_API_KEY not set, but other providers available:\n"
                f"  - {chr(10).join('  - ' + a for a in available)}\n"
                f"\nInstall litellm for multi-provider support:\n"
                f"  pip install litellm\n"
                f"\nOr set OPENAI_API_KEY for OpenAI-only mode."
            )
        
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. "
            "Please set it in a .env file or export it.\n"
            "\nFor free alternatives, consider:\n"
            "  - Google Gemini: https://makersuite.google.com/app/apikey\n"
            "  - Groq: https://console.groq.com\n"
            "  - Together AI: https://together.ai (free credits)\n"
            "\nThen install litellm: pip install litellm"
        )
    
    client = OpenAI(api_key=api_key)
    return instructor.patch(client)


def generate_evaluation_criteria(
    context: str = "birthday presents for a child",
    model: str = "gpt-4o",
    temperature: float = 0.7,
    max_retries: int = 3
) -> EvaluationCriteria:
    """
    Generate evaluation criteria for a given context using LLM.
    
    Args:
        context: The context for evaluation (e.g., "birthday presents for a 7-year-old")
        model: The OpenAI model to use
        temperature: Sampling temperature (0.0 to 2.0)
        max_retries: Maximum number of retry attempts
    
    Returns:
        EvaluationCriteria object with generated criteria
    """
    client = get_client()
    
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
            {"role": "system", "content": "You are a helpful assistant that creates structured evaluation criteria."},
            {"role": "user", "content": prompt}
        ],
        response_model=EvaluationCriteria,
    )
    
    # Set the context on the returned object
    criteria.context = context
    return criteria


def chat_with_llm(
    user_message: str,
    model: str = "gpt-4o",
    temperature: float = 0.7,
    system_prompt: Optional[str] = None
) -> str:
    """
    Simple chat interface with LLM.
    
    Args:
        user_message: The user's message
        model: The OpenAI model to use
        temperature: Sampling temperature (0.0 to 2.0)
        system_prompt: Optional system prompt
    
    Returns:
        LLM response as string
    """
    client = get_client()
    
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