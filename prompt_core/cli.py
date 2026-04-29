#!/usr/bin/env python3
"""
CLI interface for prompt-core.
"""

import json
from pathlib import Path
from typing import Optional

import typer

from .conversation_flows import (
    ConversationFlowState,
    ConversationIO,
    run_reviewed_criteria_conversation,
)
from .exceptions import (
    APIKeyError,
    ConfigFileError,
    ConfigurationError,
    CriteriaValidationError,
    ConversationFailedError,
    PromptCoreError,
    ProviderNotFoundError,
    ProviderNotSupportedError,
    TurnLimitExceededError,
)
from .models import EvaluationCriteria
from .session_logging import log_session

app = typer.Typer(help="Generate and work with evaluation criteria using LLMs")


class TyperConversationIO(ConversationIO):
    """Typer-backed I/O adapter for conversation flow functions."""

    def echo(self, message: str) -> None:
        typer.echo(message)

    def prompt(self, label: str) -> str:
        return typer.prompt(label)


def print_criteria(criteria: EvaluationCriteria, title: str) -> None:
    """Render criteria in a consistent CLI format."""
    typer.echo(f"\n{title}")
    typer.echo(f"✓ Generated {len(criteria.criteria)} criteria")
    typer.echo(f"Context: {criteria.context}")

    for i, criterion in enumerate(criteria.criteria, 1):
        typer.echo(f"\n{i}. {criterion.name} (weight: {criterion.weight})")
        typer.echo(f"   Description: {criterion.description}")
        if criterion.ideal_value:
            typer.echo(f"   Ideal: {criterion.ideal_value}")

    typer.echo("\nNormalized weights (sum to 1.0):")
    normalized = criteria.normalized_weights()
    for criterion, weight in zip(criteria.criteria, normalized):
        typer.echo(f"  {criterion.name}: {weight:.3f}")


def handle_error(error: Exception):
    """Handle errors with user-friendly messages."""
    if isinstance(error, ConfigFileError):
        typer.secho(
            f"\n✗ Configuration file error: {error.message}",
            err=True,
            fg=typer.colors.RED,
        )
    elif isinstance(error, ConfigurationError):
        typer.secho(
            f"\n✗ Configuration error: {error.message}", err=True, fg=typer.colors.RED
        )
    elif isinstance(error, APIKeyError):
        typer.secho(
            f"\n✗ API key error: {error.message}", err=True, fg=typer.colors.RED
        )
    elif isinstance(error, ProviderNotSupportedError):
        typer.secho(
            f"\n✗ Provider error: {error.message}", err=True, fg=typer.colors.RED
        )
    elif isinstance(error, ProviderNotFoundError):
        typer.secho(
            f"\n✗ Provider not found: {error.message}", err=True, fg=typer.colors.RED
        )
    elif isinstance(error, TurnLimitExceededError):
        typer.secho(f"\n✗ {error.message}", err=True, fg=typer.colors.RED)
    elif isinstance(error, ConversationFailedError):
        typer.secho(
            f"\n✗ Conversation failed: {error.message}", err=True, fg=typer.colors.RED
        )
    elif isinstance(error, CriteriaValidationError):
        typer.secho(
            f"\n✗ Validation error: {error.message}", err=True, fg=typer.colors.RED
        )
    elif isinstance(error, PromptCoreError):
        typer.secho(f"\n✗ Error: {error.message}", err=True, fg=typer.colors.RED)
    else:
        typer.secho(
            f"\n✗ Unexpected error: {str(error)[:200]}", err=True, fg=typer.colors.RED
        )
    raise typer.Exit(1)


@app.command()
def converse(
    context: str = typer.Option(
        "", "--context", "-c", help="Initial context for the conversation"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Save successful criteria to JSON file"
    ),
    max_turns: int = typer.Option(
        10,
        "--max-turns",
        "-t",
        help="Maximum number of conversation turns",
        min=1,
        max=20,
    ),
):
    """
    Interactive conversation to generate evaluation criteria.

    Example:
    prompt-core converse --context "birthday presents for child"
    """
    typer.echo(f"Starting conversation... (Ctrl+C to quit, max {max_turns} turns)")
    io = TyperConversationIO()
    flow_state = ConversationFlowState()

    try:
        if context:
            typer.echo(f"Context: {context}")

        result = run_reviewed_criteria_conversation(
            context=context,
            max_turns=max_turns,
            io=io,
            state=flow_state,
        )

        initial_result = flow_state.initial_result
        refinement_result = (
            flow_state.final_result
            if flow_state.initial_result
            and flow_state.final_result
            and flow_state.initial_result.criteria
            else None
        )

        success_judgement = False
        feedback_text: str | None = None

        if initial_result and initial_result.criteria:
            print_criteria(initial_result.criteria, title="Initial criteria:")

        if result.criteria:
            if refinement_result and refinement_result.criteria:
                print_criteria(result.criteria, title="Final criteria:")

            if output:
                with open(output, "w") as f:
                    json.dump(result.criteria.model_dump(), f, indent=2)
                typer.echo(f"\n✓ Saved to {output}")

            success_judgement = typer.confirm(
                "\nWas this experience successful?", default=True
            )

            if not success_judgement:
                feedback_text = typer.prompt("What went wrong? (optional)", default="")
                if feedback_text == "":
                    feedback_text = None

            criteria_dict = result.criteria.model_dump() if result.criteria else None

        else:
            typer.echo("\n✗ Conversation ended without generating criteria")
            success_judgement = typer.confirm(
                "\nWas this experience successful?", default=False
            )

            if not success_judgement:
                feedback_text = typer.prompt("What went wrong? (optional)", default="")
                if feedback_text == "":
                    feedback_text = None

            criteria_dict = None

        try:
            log_path = log_session(
                messages=flow_state.messages,
                criteria=criteria_dict,
                success_judgement=success_judgement,
                feedback_text=feedback_text,
                model=flow_state.model,
                turn_count=flow_state.turn_count,
                context=context,
            )
            typer.echo(f"\n📝 Session logged to: {log_path}")
        except Exception as e:
            typer.secho(f"\n⚠️  Failed to log session: {e}", fg=typer.colors.YELLOW)

        if not result.criteria:
            raise typer.Exit(1)

    except KeyboardInterrupt:
        typer.echo("\n\nConversation cancelled.")
        raise typer.Exit(0)
    except ConversationFailedError as e:
        typer.secho(
            f"\n✗ Conversation failed: {e.message}", err=True, fg=typer.colors.RED
        )
        success_judgement = typer.confirm(
            "\nWas this experience successful?", default=False
        )
        feedback_text: str | None = None
        if not success_judgement:
            feedback_text = typer.prompt("What went wrong? (optional)", default="")
            if feedback_text == "":
                feedback_text = None

        log_path = log_session(
            messages=flow_state.messages,
            criteria=None,
            success_judgement=success_judgement,
            feedback_text=feedback_text,
            model=flow_state.model,
            turn_count=flow_state.turn_count,
            context=context,
        )
        typer.echo(f"\n📝 Session logged to: {log_path}")
        raise typer.Exit(1)
    except TurnLimitExceededError as e:
        typer.secho(f"\n✗ {e.message}", err=True, fg=typer.colors.RED)

        log_path = log_session(
            messages=flow_state.messages,
            criteria=None,
            success_judgement=False,
            feedback_text="Turn limit reached",
            model=flow_state.model,
            turn_count=flow_state.turn_count,
            context=context,
        )
        typer.echo(f"\n📝 Session logged to: {log_path}")
        raise typer.Exit(1)
    except Exception as e:
        handle_error(e)


def main():
    """Entry point for the CLI application."""
    app()


if __name__ == "__main__":
    main()
