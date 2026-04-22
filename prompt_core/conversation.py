from typing import List, Optional, Literal
from pydantic import BaseModel, Field, model_validator


class ConversationAction(BaseModel):
    """LLM's decision about conversation flow with discriminator field."""
    action: Literal["continue", "success", "failure"]
    message: Optional[str] = None  # Required for "continue" and "failure" actions
    criteria: Optional["EvaluationCriteria"] = None  # Required for "success" action
    
    @model_validator(mode='after')
    def validate_action_consistency(self):
        """Validate that the correct fields are provided based on action."""
        if self.action in ["continue", "failure"] and not self.message:
            raise ValueError(f"{self.action} action requires message")
        if self.action == "success" and not self.criteria:
            raise ValueError("success action requires criteria")
        return self


class ConversationResult(BaseModel):
    """Result of a conversation turn."""
    criteria: Optional["EvaluationCriteria"] = None
    message: str  # Message to show user
    is_complete: bool  # True if conversation ended (success or failure)
    
    @classmethod
    def continuing(cls, message: str) -> "ConversationResult":
        return cls(criteria=None, message=message, is_complete=False)
    
    @classmethod
    def success(cls, criteria: "EvaluationCriteria") -> "ConversationResult":
        return cls(criteria=criteria, message="Criteria generated successfully!", is_complete=True)
    
    @classmethod
    def failure(cls, message: str) -> "ConversationResult":
        return cls(criteria=None, message=f"Failed: {message}", is_complete=True)


class ConversationOrchestrator:
    """Manages conversation state and LLM interactions for criteria generation."""
    
    def __init__(self, initial_context: str = "", max_turns: int = 10, model: str = "gpt-4o-mini"):
        self.messages = []
        self.turn_count = 0
        self.max_turns = max_turns
        self.model = model
        
        # Minimal system prompt - instructor will handle schema explanation
        system_prompt = """
        You are a helpful assistant that guides users through the process of creating structured information.
        You will have a multi-turn conversation to gather the information.
        Ask one question at a time to avoid overwhelming the user.
        Be empathetic and helpful.
        
        You're helping the user create evaluation criteria for decision making.
        You need to ask questions to understand what criteria they want to evaluate options by.
        
        Important requirements:
        1. The final criteria list must include at least 2 criteria
        2. It MUST include a criterion named "budget" (case-insensitive)
        
        Your conversation can have three outcomes:
        1. "continue": Ask another question to gather more information
        2. "success": Return valid evaluation criteria that meets all requirements
        3. "failure": Give up if the conversation isn't productive
        
        Format your response according to the ConversationAction schema.
        """
        
        self.messages.append({"role": "system", "content": system_prompt})
        
        if initial_context:
            self.messages.append({
                "role": "user", 
                "content": f"I'd like to create evaluation criteria for: {initial_context}"
            })
    
    def process_turn(self, user_input: str) -> ConversationResult:
        """
        Process one conversation turn.
        Returns: ConversationResult object with outcome
        
        Logic:
        1. Check turn limit (max 10) - raise ValueError if exceeded
        2. Add user message to history
        3. Call LLM via instructor with max_retries=3
        4. Handle ConversationAction:
            - "continue": Return ConversationResult.continuing(message)
            - "success": Return ConversationResult.success(criteria)
            - "failure": Raise ValueError with message
        5. Exceptions propagate (fail fast)
        """
        # Check turn limit
        if self.turn_count >= self.max_turns:
            raise ValueError(f"Maximum conversation turns ({self.max_turns}) reached")
        
        # Add user message to history
        if user_input.strip():
            self.messages.append({"role": "user", "content": user_input})
        
        self.turn_count += 1
        
        # Call LLM - exceptions will propagate
        action = self._call_llm()
        
        # Add assistant message to history if needed
        if action.message:
            self.messages.append({"role": "assistant", "content": action.message})
        
        # Handle action
        if action.action == "continue":
            return ConversationResult.continuing(action.message)
        elif action.action == "success":
            return ConversationResult.success(action.criteria)
        elif action.action == "failure":
            raise ValueError(f"LLM indicated failure: {action.message}")
        else:
            # Should not happen due to Literal type
            raise ValueError(f"Invalid action received: {action.action}")
    
    def _call_llm(self) -> ConversationAction:
        """Call LLM using provider-agnostic interface."""
        from .llm_provider import get_provider
        
        try:
            provider = get_provider()
            return provider.create_structured_response(
                model=self.model,
                messages=self.messages,
                response_model=ConversationAction,
                max_retries=3
            )
        except ImportError as e:
            # If no providers are available, give helpful error
            raise ImportError(
                f"No LLM providers available. {e}\n"
                "Install at least one provider:\n"
                "  - OpenAI: pip install openai\n"
                "  - Or install other provider packages"
            )
    
    
    
    


# Import here to avoid circular import
from .models import EvaluationCriteria

# Update forward references
ConversationAction.model_rebuild()
ConversationResult.model_rebuild()