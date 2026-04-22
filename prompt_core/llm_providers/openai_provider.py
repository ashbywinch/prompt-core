"""
OpenAI provider implementation without instructor dependency.
"""
import json
import os
from typing import Any, Dict, List, Optional, Type, TypeVar
import logging

from ..llm_provider import LLMProvider, ValidationError, AuthenticationError, ConnectionError

T = TypeVar('T')

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """
    OpenAI provider using direct API calls without instructor.
    
    This implementation manually handles structured responses by:
    1. Generating a schema from Pydantic model
    2. Asking LLM to output JSON matching the schema
    3. Validating and parsing the response
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            base_url: Custom base URL for OpenAI-compatible API
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. "
                "Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )
        
        self.base_url = base_url
        self._client = None
        
    @property
    def client(self):
        """Lazy-load OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
            except ImportError:
                raise ImportError(
                    "OpenAI Python SDK not installed. "
                    "Install with: pip install openai"
                )
        return self._client
    
    def create_structured_response(
        self,
        model: str,
        messages: List[Dict[str, str]],
        response_model: Type[T],
        max_retries: int = 3,
        temperature: float = 0.7,
        **kwargs
    ) -> T:
        """
        Generate structured response using OpenAI API.
        
        This method manually creates a JSON schema from the Pydantic model
        and asks the LLM to output JSON matching that schema.
        """
        from pydantic import BaseModel
        
        if not issubclass(response_model, BaseModel):
            raise TypeError(f"response_model must be a Pydantic BaseModel, got {type(response_model)}")
        
        # Create JSON schema from Pydantic model
        schema = response_model.model_json_schema()
        
        # Add explicit JSON formatting instructions to system message
        enhanced_messages = self._add_json_formatting(messages, schema, response_model.__name__)
        
        # Try up to max_retries times
        for attempt in range(max_retries + 1):  # +1 for initial attempt
            try:
                response = self._call_openai(
                    model=model,
                    messages=enhanced_messages,
                    temperature=temperature,
                    **kwargs
                )
                
                # Parse and validate response
                result = self._parse_and_validate(response, response_model)
                return result
                
            except (ValidationError, json.JSONDecodeError) as e:
                if attempt < max_retries:
                    logger.debug(f"Validation failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    # Add error feedback to messages for next attempt
                    feedback = f"The response failed validation: {str(e)}. Please try again."
                    enhanced_messages.append({"role": "user", "content": feedback})
                    continue
                else:
                    raise ValidationError(
                        f"Failed to get valid response after {max_retries + 1} attempts: {e}"
                    )
            except Exception as e:
                # Re-raise non-validation errors immediately
                if "authentication" in str(e).lower() or "api key" in str(e).lower():
                    raise AuthenticationError(f"OpenAI authentication failed: {e}")
                elif "connection" in str(e).lower() or "timeout" in str(e).lower():
                    raise ConnectionError(f"OpenAI connection failed: {e}")
                else:
                    raise
    
    def _add_json_formatting(
        self, 
        messages: List[Dict[str, str]], 
        schema: Dict[str, Any],
        model_name: str
    ) -> List[Dict[str, str]]:
        """Add JSON formatting instructions to messages."""
        # Create a copy of messages
        enhanced_messages = messages.copy()
        
        # Check if there's a system message
        has_system = any(msg.get("role") == "system" for msg in enhanced_messages)
        
        # Create JSON formatting instructions
        json_instructions = f"""
You MUST respond with a valid JSON object that matches this schema:

```json
{json.dumps(schema, indent=2)}
```

The JSON MUST represent a valid instance of the {model_name} model.
Respond ONLY with the JSON object, no other text.
"""
        
        if has_system:
            # Append to existing system message
            for i, msg in enumerate(enhanced_messages):
                if msg.get("role") == "system":
                    enhanced_messages[i]["content"] += "\n\n" + json_instructions
                    break
        else:
            # Add system message at the beginning
            enhanced_messages.insert(0, {"role": "system", "content": json_instructions})
        
        return enhanced_messages
    
    def _call_openai(self, model: str, messages: List[Dict[str, str]], **kwargs) -> str:
        """Make OpenAI API call."""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            
            content = response.choices[0].message.content
            if content is None:
                raise ValidationError("LLM returned empty response")
            
            return content.strip()
            
        except Exception as e:
            # Map OpenAI exceptions to our abstract exceptions
            error_msg = str(e).lower()
            if "authentication" in error_msg or "api key" in error_msg:
                raise AuthenticationError(f"OpenAI authentication failed: {e}")
            elif "connection" in error_msg or "timeout" in error_msg:
                raise ConnectionError(f"OpenAI connection failed: {e}")
            else:
                raise
    
    def _parse_and_validate(self, response_text: str, response_model: Type[T]) -> T:
        """Parse JSON response and validate against Pydantic model."""
        # Extract JSON from response (in case LLM added markdown or other formatting)
        json_text = self._extract_json(response_text)
        
        try:
            # Parse JSON
            data = json.loads(json_text)
            
            # Validate using Pydantic model
            return response_model(**data)
            
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON: {e}\nResponse was: {response_text}")
        except Exception as e:
            raise ValidationError(f"Validation failed: {e}\nData was: {data}")
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from text that might contain markdown or other formatting."""
        # Try to find JSON object or array
        import re
        
        # Look for ```json ... ``` code blocks
        json_block = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if json_block:
            return json_block.group(1).strip()
        
        # Look for JSON object {...}
        json_obj = re.search(r'\{[\s\S]*\}', text)
        if json_obj:
            return json_obj.group(0).strip()
        
        # Look for JSON array [...]
        json_array = re.search(r'\[[\s\S]*\]', text)
        if json_array:
            return json_array.group(0).strip()
        
        # If no JSON found, return original text (might be plain JSON)
        return text.strip()


# Convenience function for backward compatibility
def get_openai_provider(**kwargs) -> OpenAIProvider:
    """Get configured OpenAI provider."""
    return OpenAIProvider(**kwargs)