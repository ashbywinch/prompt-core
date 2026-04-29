from __future__ import annotations

import json
from typing import Literal, Optional

from pydantic import BaseModel, model_validator

from .models import EvaluationCriteria


class ConversationAction(BaseModel):
    action: Literal["continue", "success", "failure"]
    message: Optional[str] = None
    criteria: Optional["EvaluationCriteria"] = None

    @model_validator(mode="after")
    def validate_action_consistency(self):
        if self.action in ["continue", "failure"] and not self.message:
            raise ValueError(f"{self.action} action requires message")
        if self.action == "success" and not self.criteria:
            raise ValueError("success action requires criteria")
        return self


class ConversationResult(BaseModel):
    criteria: Optional["EvaluationCriteria"] = None
    message: str
    is_complete: bool

    @classmethod
    def continuing(cls, message: str) -> "ConversationResult":
        return cls(criteria=None, message=message, is_complete=False)

    @classmethod
    def success(cls, criteria: "EvaluationCriteria") -> "ConversationResult":
        return cls(
            criteria=criteria,
            message="Criteria generated successfully!",
            is_complete=True,
        )

    @classmethod
    def failure(cls, message: str) -> "ConversationResult":
        return cls(criteria=None, message=f"Failed: {message}", is_complete=True)


class CriteriaRefinementAction(BaseModel):
    action: Literal["continue", "success", "failure"]
    message: Optional[str] = None
    criteria: Optional["EvaluationCriteria"] = None

    @model_validator(mode="after")
    def validate_action_consistency(self):
        if self.action in ["continue", "failure"] and not self.message:
            raise ValueError(f"{self.action} action requires message")
        if self.action == "success" and not self.criteria:
            raise ValueError("success action requires criteria")
        return self


class CriteriaRefinementOrchestrator:
    def __init__(self, initial_criteria: EvaluationCriteria, max_turns: int = 5):
        from prompt_core.config import config

        self.initial_criteria = initial_criteria
        self.messages = []
        self.turn_count = 0
        self.max_turns = max_turns
        self.model = config.model

        criteria_json = json.dumps(initial_criteria.model_dump(), indent=2)
        system_prompt = f"""
        You are a helpful assistant running a refinement conversation for existing evaluation criteria.
        You will have a multi-turn conversation (maximum {self.max_turns} turns) with the user.

        YOUR ROLE:
        - Discuss the current criteria with the user
        - Understand whether they want to keep the criteria as-is or update them
        - Ask one question at a time if their requested changes are unclear
        - End with a final criteria object when the user is satisfied

        RESPONSE ACTIONS:
        - action="continue": ask follow-up question or summarize what you're changing next
        - action="success": return the final criteria object (either unchanged or updated)
        - action="failure": use only if the user refuses to engage and no usable final criteria can be produced

        IMPORTANT:
        - Conversation limit: {self.max_turns} turns total
        - Base the final criteria only on user-provided feedback
        - Preserve the same context unless user asks to change it
        - Final criteria MUST include a criterion named exactly "budget" (case-insensitive)
        """

        self.messages.append({"role": "system", "content": system_prompt})
        self.messages.append(
            {
                "role": "user",
                "content": (
                    "Here is the current criteria to review:\n"
                    f"{criteria_json}\n"
                    "Please discuss updates with me and return final criteria when ready."
                ),
            }
        )

    def process_turn(self, user_input: str) -> ConversationResult:
        if self.turn_count >= self.max_turns:
            from prompt_core.exceptions import TurnLimitExceededError

            raise TurnLimitExceededError(self.max_turns)

        if user_input.strip():
            self.messages.append({"role": "user", "content": user_input})

        self.turn_count += 1
        action = self._call_llm()

        if action.message:
            self.messages.append({"role": "assistant", "content": action.message})

        if action.action == "continue":
            return ConversationResult.continuing(action.message)
        if action.action == "success":
            return ConversationResult(
                criteria=action.criteria,
                message=action.message or "Criteria finalized successfully!",
                is_complete=True,
            )
        if action.action == "failure":
            from prompt_core.exceptions import ConversationFailedError

            raise ConversationFailedError(action.message)

        from prompt_core.exceptions import InvalidResponseError

        raise InvalidResponseError(f"Invalid action received: {action.action}")

    def _call_llm(self) -> CriteriaRefinementAction:
        from prompt_core.config import config
        from prompt_core.exceptions import ProviderNotFoundError
        from prompt_core.llm_interaction import get_client

        try:
            client = get_client(supports_tools=config.model_supports_tools)
            return client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                response_model=CriteriaRefinementAction,
                max_retries=config.max_retries,
                timeout=config.request_timeout_seconds,
            )
        except ImportError as e:
            raise ProviderNotFoundError(
                f"No LLM providers available. {e}\n"
                "Install litellm for multi-provider LLM support: uv add litellm"
            )


class ConversationOrchestrator:
    def __init__(self, initial_context: str = "", max_turns: int = 10):
        from prompt_core.config import config

        self.messages = []
        self.turn_count = 0
        self.max_turns = max_turns
        self.model = config.model

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

        self.messages.append({"role": "system", "content": system_prompt})

        if initial_context:
            self.messages.append(
                {
                    "role": "user",
                    "content": f"I'd like to create evaluation criteria for: {initial_context}",
                }
            )

    def process_turn(self, user_input: str) -> ConversationResult:
        if self.turn_count >= self.max_turns:
            from prompt_core.exceptions import TurnLimitExceededError

            raise TurnLimitExceededError(self.max_turns)

        if user_input.strip():
            self.messages.append({"role": "user", "content": user_input})

        self.turn_count += 1
        action = self._call_llm()

        if action.message:
            self.messages.append({"role": "assistant", "content": action.message})

        if action.action == "continue":
            return ConversationResult.continuing(action.message)
        if action.action == "success":
            return ConversationResult.success(action.criteria)
        if action.action == "failure":
            from prompt_core.exceptions import ConversationFailedError

            raise ConversationFailedError(action.message)

        from prompt_core.exceptions import InvalidResponseError

        raise InvalidResponseError(f"Invalid action received: {action.action}")

    def _call_llm(self) -> ConversationAction:
        from prompt_core.config import config
        from prompt_core.exceptions import ProviderNotFoundError
        from prompt_core.llm_interaction import get_client

        try:
            client = get_client(supports_tools=config.model_supports_tools)
            return client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                response_model=ConversationAction,
                max_retries=config.max_retries,
                timeout=config.request_timeout_seconds,
            )
        except ImportError as e:
            raise ProviderNotFoundError(
                f"No LLM providers available. {e}\n"
                "Install litellm for multi-provider LLM support: uv add litellm"
            )


ConversationAction.model_rebuild()
ConversationResult.model_rebuild()
CriteriaRefinementAction.model_rebuild()
