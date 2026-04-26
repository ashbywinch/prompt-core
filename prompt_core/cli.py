#!/usr/bin/env python3
"""
CLI interface for prompt-core.
"""

import json
from pathlib import Path
from typing import Optional

import typer

from .conversation import ConversationOrchestrator
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

app = typer.Typer(help="Generate and work with evaluation criteria using LLMs")


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

            # Show assistant response
            typer.echo(f"\nAssistant: {result.message}")

            # Check if we should continue
            if result.is_complete:
                break

        # Handle result
        if result.criteria:
            typer.echo(f"\n✓ Generated {len(result.criteria.criteria)} criteria")
            typer.echo(f"Context: {result.criteria.context}")

            # Display criteria
            for i, criterion in enumerate(result.criteria.criteria, 1):
                typer.echo(f"\n{i}. {criterion.name} (weight: {criterion.weight})")
                typer.echo(f"   Description: {criterion.description}")
                if criterion.ideal_value:
                    typer.echo(f"   Ideal: {criterion.ideal_value}")

            # Save to file if requested
            if output:
                with open(output, "w") as f:
                    json.dump(result.criteria.model_dump(), f, indent=2)
                typer.echo(f"\n✓ Saved to {output}")

            # Show normalized weights
            typer.echo("\nNormalized weights (sum to 1.0):")
            normalized = result.criteria.normalized_weights()
            for criterion, weight in zip(result.criteria.criteria, normalized):
                typer.echo(f"  {criterion.name}: {weight:.3f}")

        else:
            # Failure case
            typer.echo("\n✗ Conversation ended without generating criteria")
            raise typer.Exit(1)

    except KeyboardInterrupt:
        typer.echo("\n\nConversation cancelled.")
        raise typer.Exit(0)
    except Exception as e:
        handle_error(e)


def main():
    """Entry point for the CLI application."""
    app()


if __name__ == "__main__":
    main()
