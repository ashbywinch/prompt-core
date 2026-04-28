#!/usr/bin/env python3
"""
CLI interface for prompt-core.
"""

import json
from pathlib import Path
from typing import Optional

import typer

from .conversation import ConversationOrchestrator, CriteriaRefinementOrchestrator
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
    orchestrator: ConversationOrchestrator | None = None
    refinement_orchestrator: CriteriaRefinementOrchestrator | None = None

    try:
        orchestrator = ConversationOrchestrator(
            initial_context=context, max_turns=max_turns
        )

        # Start conversation
        if context:
            typer.echo(f"Context: {context}")
            result = orchestrator.process_turn("Let's begin.")
        else:
            result = orchestrator.process_turn("")

        # Show initial response
        typer.echo(f"\nAssistant: {result.message}")

        # Interactive loop
        while not result.is_complete:
            # Get user input
            user_input = typer.prompt("\nYou")

            # Process turn
            result = orchestrator.process_turn(user_input)

            typer.echo(f"\nAssistant: {result.message}")

            if result.is_complete:
                break

        success_judgement = False
        feedback_text: str | None = None

        all_messages = list(orchestrator.messages)

        if result.criteria:
            print_criteria(result.criteria, title="Initial criteria:")

            typer.echo(
                "\nAssistant: What do you think about these criteria? We can keep them or update them."
            )
            refinement_orchestrator = CriteriaRefinementOrchestrator(
                initial_criteria=result.criteria, max_turns=max_turns
            )

            refinement_result = refinement_orchestrator.process_turn(
                typer.prompt("\nYou")
            )
            typer.echo(f"\nAssistant: {refinement_result.message}")

            while not refinement_result.is_complete:
                refinement_result = refinement_orchestrator.process_turn(
                    typer.prompt("\nYou")
                )
                typer.echo(f"\nAssistant: {refinement_result.message}")

            result = refinement_result
            all_messages.extend(refinement_orchestrator.messages)

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
                messages=all_messages,
                criteria=criteria_dict,
                success_judgement=success_judgement,
                feedback_text=feedback_text,
                model=orchestrator.model,
                turn_count=len([m for m in all_messages if m["role"] == "user"]),
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

        combined_messages: list[dict[str, str]] = []
        model = "unknown"
        turn_count = 0

        if orchestrator is not None:
            combined_messages.extend(orchestrator.messages)
            model = orchestrator.model
            turn_count += orchestrator.turn_count

        if refinement_orchestrator is not None:
            combined_messages.extend(refinement_orchestrator.messages)
            model = refinement_orchestrator.model
            turn_count += refinement_orchestrator.turn_count

        log_path = log_session(
            messages=combined_messages,
            criteria=None,
            success_judgement=success_judgement,
            feedback_text=feedback_text,
            model=model,
            turn_count=turn_count,
            context=context,
        )
        typer.echo(f"\n📝 Session logged to: {log_path}")
        raise typer.Exit(1)
    except TurnLimitExceededError as e:
        typer.secho(f"\n✗ {e.message}", err=True, fg=typer.colors.RED)

        combined_messages: list[dict[str, str]] = []
        model = "unknown"
        turn_count = 0

        if orchestrator is not None:
            combined_messages.extend(orchestrator.messages)
            model = orchestrator.model
            turn_count += orchestrator.turn_count

        if refinement_orchestrator is not None:
            combined_messages.extend(refinement_orchestrator.messages)
            model = refinement_orchestrator.model
            turn_count += refinement_orchestrator.turn_count

        log_path = log_session(
            messages=combined_messages,
            criteria=None,
            success_judgement=False,
            feedback_text="Turn limit reached",
            model=model,
            turn_count=turn_count,
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
