from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional
from datetime import datetime, timezone

from pydantic import BaseModel, model_validator


@dataclass
class Message:
    """A single message in the conversation history."""

    role: Literal["system", "user", "assistant"]
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for LLM API calls."""
        return {"role": self.role, "content": self.content}


@dataclass
class LLMResponse:
    """Wraps a parsed LLM response with metadata."""

    content: ConversationAction  # The parsed ConversationAction
    model: str = ""
    usage: dict[str, Any] | None = None
    cost: float | None = None


class ConversationAction(BaseModel):
    """LLM's decision about conversation flow with discriminator field."""

    action: Literal["continue", "success", "failure"]
    message: Optional[str] = None  # Required for "continue" and "failure" actions
    criteria: Optional[BaseModel] = None  # Required for "success" action

    @model_validator(mode="after")
    def validate_action_consistency(self):
        """Validate that the correct fields are provided based on action."""
        if self.action in ["continue", "failure"] and not self.message:
            raise ValueError(f"{self.action} action requires message")
        if self.action == "success" and not self.criteria:
            raise ValueError("success action requires criteria")
        return self


class ConversationResult(BaseModel):
    """Result of a conversation turn."""

    criteria: Optional[BaseModel] = None
    message: str  # Message to show user
    is_complete: bool  # True if conversation ended (success or failure)

    @classmethod
    def continuing(cls, message: str) -> "ConversationResult":
        return cls(criteria=None, message=message, is_complete=False)

    @classmethod
    def success(cls, criteria: BaseModel) -> "ConversationResult":
        return cls(
            criteria=criteria,
            message="Criteria generated successfully!",
            is_complete=True,
        )

    @classmethod
    def failure(cls, message: str) -> "ConversationResult":
        return cls(criteria=None, message=f"Failed: {message}", is_complete=True)


class ConversationOrchestrator:
    """Manages conversation state and LLM interactions for criteria generation."""

    def __init__(self, initial_context: str = "", max_turns: int = 10):
        from .config import config

        self.messages: list[Message] = []
        self.turn_count = 0
        self.max_turns = max_turns
        self.model = config.model

        # System prompt - focus on behavioral guidance, instructor handles schema
        system_prompt = f"""
        You are a helpful assistant that guides users through a structured conversation to produce the specified output.
        You will have a multi-turn conversation (maximum {self.max_turns} turns) to gather information.
        
        YOUR ROLE:
        Guide the conversation to gather enough information from the user to produce the best possible output, from information they provide to you, in the required format. 
        At the end of the conversation we want the user to think that the output is much better than what they would have come up with on their own.
        Ask questions one at a time. Start very open-ended and get more specific if you need to.
        
        KEY PRINCIPLES:
        - Base your final response only on things the user explicitly tells you
        - Ask for clarification if responses are unclear
        - Acknowledge answers before asking follow-up questions
        - Recognize when you have enough comprehensive information and can finish the conversation
        
        HANDLING DIFFICULT CONVERSATIONS:
        If the user is consistently vague or unresponsive:
        1. First, try asking more specific questions to help them
        2. If they haven't provided the required information after the maximum number of turns, end the conversation with the failure message
        3. Provide a helpful explanation of why the conversation failed
        
        EXAMPLE SCENARIOS:
        
        Productive conversation to generate a person object with name and age:
        - You: "Can you tell me you name?"
        - User: "John"
        - You: "And how old are you?"
        - User: "I don't know!"
        - You: "Well, when's your birthday?"
        - User: "1st April"
        - You: "And what year were you born?"
        - User: "1926"
        - You: "Looks like you're 100 years old. Does that sound about right?"
        - User: "Yes!"
        - You: [Return the person object with the provided name and age]
        
        Unproductive conversation to generate a person object with a name and age:
        - You: "Can you tell me your name?"
        - User: "I'm not sure"
        - You: "Do you know how old you are?"
        - User: "No"
        - You: "How about what year you were born in?"
        - User: "No idea"
        - You: [End the conversation with brief explanation about needing the name and age]
        
        BAD EXAMPLE of a conversation to generate a holiday destination:
        - You: "Tell me about some memorable holidays and what you liked about them"
        - User: "Go away"
        - User: [ continues not to engage at all ]
        - You: [ End the conversation with a success response, randomly suggesting they go to Portugal ]

        IMPORTANT:
        - Conversation limit: {self.max_turns} turns total
        - Focus on gathering real, specific information
        - Follow the structured response format provided
        - You MUST include a criterion named exactly "budget" (case-insensitive) in the criteria list
        """

        self.messages.append(Message(role="system", content=system_prompt))

        if initial_context:
            self.messages.append(
                Message(
                    role="user",
                    content=f"I'd like to create evaluation criteria for: {initial_context}",
                )
            )

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
            from .exceptions import TurnLimitExceededError

            raise TurnLimitExceededError(self.max_turns)

        # Add user message to history
        if user_input.strip():
            self.messages.append(Message(role="user", content=user_input))

        self.turn_count += 1

        # Call LLM - exceptions will propagate
        llm_response = self._call_llm()
        action = llm_response.content

        # Add assistant message to history if needed
        if action.message:
            self.messages.append(Message(role="assistant", content=action.message))

        # Handle action
        if action.action == "continue":
            return ConversationResult.continuing(action.message)
        elif action.action == "success":
            return ConversationResult.success(action.criteria)
        elif action.action == "failure":
            from .exceptions import ConversationFailedError

            raise ConversationFailedError(action.message)
        else:
            # Should not happen due to Literal type
            from .exceptions import InvalidResponseError

            raise InvalidResponseError(f"Invalid action received: {action.action}")

    def _call_llm(self) -> LLMResponse:
        """Call LLM using instructor with multi-provider support.

        Returns:
            LLMResponse containing the parsed ConversationAction and usage metadata.
        """
        import litellm
        from .config import config
        from .exceptions import ProviderNotFoundError
        from .llm_interaction import get_client

        try:
            client = get_client(supports_tools=config.model_supports_tools)

            # Use create_with_completion to capture the raw response with usage
            parsed, raw_response = client.chat.completions.create_with_completion(
                model=self.model,
                messages=[m.to_dict() for m in self.messages],
                response_model=ConversationAction,
                max_retries=config.max_retries,
                timeout=config.request_timeout_seconds,
            )

            usage = None
            cost = None
            if raw_response and raw_response.usage:
                usage = (
                    raw_response.usage.model_dump()
                    if hasattr(raw_response.usage, "model_dump")
                    else raw_response.usage
                )
                try:
                    cost = litellm.completion_cost(completion_response=raw_response)
                except Exception:
                    cost = None

            return LLMResponse(
                content=parsed,
                model=self.model,
                usage=usage,
                cost=cost,
            )
        except ImportError as e:
            # If no providers are available, give helpful error
            raise ProviderNotFoundError(
                f"No LLM providers available. {e}\n"
                "Install litellm for multi-provider LLM support: uv add litellm"
            )

    def run_conversation(self, user_inputs: list[str]) -> ConversationResult:
        """Run through a sequence of user inputs until the conversation completes.

        Each input is processed as a separate turn. Returns the final
        ConversationResult once the conversation ends (success, failure,
        or turn limit). Raises if the conversation does not complete
        within the available turns or the provided inputs.
        """
        from .exceptions import TurnLimitExceededError

        for user_input in user_inputs:
            try:
                result = self.process_turn(user_input)
            except TurnLimitExceededError:
                raise TurnLimitExceededError(
                    f"Conversation did not complete within {self.max_turns} turns "
                    f"across {len(user_inputs)} provided inputs"
                )

            if result.is_complete:
                return result

        raise ValueError(
            f"Conversation did not complete after {len(user_inputs)} inputs "
            f"(max_turns={self.max_turns})"
        )


# Update forward references
ConversationAction.model_rebuild()
ConversationResult.model_rebuild()
