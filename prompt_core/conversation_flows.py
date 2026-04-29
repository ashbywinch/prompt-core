"""Function-style composition for conversation flows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .conversation import (
    ConversationOrchestrator,
    ConversationResult,
    CriteriaRefinementOrchestrator,
)
from .models import EvaluationCriteria


class ConversationIO(Protocol):
    """Minimal I/O contract for interactive conversations."""

    def echo(self, message: str) -> None:
        """Display one message to the user."""

    def prompt(self, label: str) -> str:
        """Read one user message."""


class ConversationOrchestratorLike(Protocol):
    """Shared shape used by the flow helper functions."""

    messages: list[dict[str, str]]
    turn_count: int
    model: str

    def process_turn(self, user_input: str) -> ConversationResult:
        """Process one turn and return the current result."""


@dataclass
class ConversationFlowState:
    """Mutable state captured while running composed conversation functions."""

    messages: list[dict[str, str]] = field(default_factory=list)
    model: str = "unknown"
    turn_count: int = 0
    initial_result: ConversationResult | None = None
    final_result: ConversationResult | None = None


def _record_orchestrator(
    state: ConversationFlowState,
    orchestrator: ConversationOrchestratorLike,
) -> None:
    state.messages.extend(orchestrator.messages)
    state.turn_count += orchestrator.turn_count
    state.model = orchestrator.model


def run_orchestrator_chat(
    orchestrator: ConversationOrchestratorLike,
    first_user_input: str,
    io: ConversationIO,
) -> ConversationResult:
    """Run one orchestrator conversation until it reaches completion."""
    result = orchestrator.process_turn(first_user_input)
    io.echo(f"\nAssistant: {result.message}")

    while not result.is_complete:
        user_input = io.prompt("\nYou")
        result = orchestrator.process_turn(user_input)
        io.echo(f"\nAssistant: {result.message}")

    return result


def run_initial_criteria_conversation(
    context: str,
    max_turns: int,
    io: ConversationIO,
    state: ConversationFlowState,
) -> ConversationResult:
    """Run the base criteria conversation and return its result."""
    orchestrator = ConversationOrchestrator(
        initial_context=context, max_turns=max_turns
    )

    try:
        first_input = "Let's begin." if context else ""
        result = run_orchestrator_chat(
            orchestrator=orchestrator,
            first_user_input=first_input,
            io=io,
        )
        state.initial_result = result
        return result
    finally:
        _record_orchestrator(state, orchestrator)


def run_refinement_conversation(
    initial_criteria: EvaluationCriteria,
    max_turns: int,
    io: ConversationIO,
    state: ConversationFlowState,
) -> ConversationResult:
    """Run post-result refinement chat and return final criteria result."""
    io.echo(
        "\nAssistant: What do you think about these criteria? "
        "We can keep them or update them."
    )
    orchestrator = CriteriaRefinementOrchestrator(
        initial_criteria=initial_criteria,
        max_turns=max_turns,
    )

    try:
        result = run_orchestrator_chat(
            orchestrator=orchestrator,
            first_user_input=io.prompt("\nYou"),
            io=io,
        )
        return result
    finally:
        _record_orchestrator(state, orchestrator)


def run_reviewed_criteria_conversation(
    context: str,
    max_turns: int,
    io: ConversationIO,
    state: ConversationFlowState,
) -> ConversationResult:
    """Compose base criteria chat with optional refinement chat."""
    initial_result = run_initial_criteria_conversation(
        context=context,
        max_turns=max_turns,
        io=io,
        state=state,
    )

    if initial_result.criteria:
        final_result = run_refinement_conversation(
            initial_criteria=initial_result.criteria,
            max_turns=max_turns,
            io=io,
            state=state,
        )
    else:
        final_result = initial_result

    state.final_result = final_result
    return final_result
