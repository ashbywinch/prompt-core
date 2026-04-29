"""EvaluationCriteria-specific conversation functions with minimal boilerplate."""

from __future__ import annotations

from prompt_core.conversation_runtime import ConversationTools, conversation_function

from .conversation import (
    ConversationOrchestrator,
    ConversationResult,
    CriteriaRefinementOrchestrator,
)
from .presentation import print_criteria


@conversation_function
def run_initial_criteria_conversation(
    context: str,
    max_turns: int,
    *,
    tools: ConversationTools,
) -> ConversationResult:
    orchestrator = ConversationOrchestrator(initial_context=context, max_turns=max_turns)
    first_input = "Let's begin." if context else ""
    return tools.chat(orchestrator=orchestrator, first_user_input=first_input)


@conversation_function
def run_reviewed_criteria_conversation(
    context: str,
    max_turns: int,
    *,
    tools: ConversationTools,
) -> ConversationResult:
    initial_result = run_initial_criteria_conversation(
        context=context,
        max_turns=max_turns,
        tools=tools,
    )

    if not initial_result.criteria:
        return initial_result

    print_criteria(
        criteria=initial_result.criteria,
        title="Initial criteria:",
        echo=tools.io.echo,
    )

    tools.io.echo(
        "\nAssistant: What do you think about these criteria? "
        "We can keep them or update them."
    )
    orchestrator = CriteriaRefinementOrchestrator(
        initial_criteria=initial_result.criteria,
        max_turns=max_turns,
    )
    final_result = tools.chat(
        orchestrator=orchestrator,
        first_user_input=tools.io.prompt("\nYou"),
    )
    if final_result.criteria:
        print_criteria(
            criteria=final_result.criteria,
            title="Final criteria:",
            echo=tools.io.echo,
        )
    return final_result
