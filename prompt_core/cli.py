#!/usr/bin/env python3
"""
CLI interface for prompt-core.
"""
import typer
import json
from typing import Optional
from pathlib import Path

from . import generate_evaluation_criteria, chat_with_llm, EvaluationCriteria
from .conversation import ConversationOrchestrator
from .exceptions import (
    PromptCoreError, ConfigurationError, ConfigFileError, APIKeyError,
    ProviderNotSupportedError, ProviderNotFoundError, TurnLimitExceededError,
    ConversationFailedError, CriteriaValidationError
)

app = typer.Typer(help="Generate and work with evaluation criteria using LLMs")


def handle_error(error: Exception):
    """Handle errors with user-friendly messages."""
    if isinstance(error, ConfigFileError):
        typer.echo(f"\n✗ Configuration file error: {error.message}", err=True, fg=typer.colors.RED)
    elif isinstance(error, ConfigurationError):
        typer.echo(f"\n✗ Configuration error: {error.message}", err=True, fg=typer.colors.RED)
    elif isinstance(error, APIKeyError):
        typer.echo(f"\n✗ API key error: {error.message}", err=True, fg=typer.colors.RED)
    elif isinstance(error, ProviderNotSupportedError):
        typer.echo(f"\n✗ Provider error: {error.message}", err=True, fg=typer.colors.RED)
    elif isinstance(error, ProviderNotFoundError):
        typer.echo(f"\n✗ Provider not found: {error.message}", err=True, fg=typer.colors.RED)
    elif isinstance(error, TurnLimitExceededError):
        typer.echo(f"\n✗ {error.message}", err=True, fg=typer.colors.RED)
    elif isinstance(error, ConversationFailedError):
        typer.echo(f"\n✗ Conversation failed: {error.message}", err=True, fg=typer.colors.RED)
    elif isinstance(error, CriteriaValidationError):
        typer.echo(f"\n✗ Validation error: {error.message}", err=True, fg=typer.colors.RED)
    elif isinstance(error, PromptCoreError):
        typer.echo(f"\n✗ Error: {error.message}", err=True, fg=typer.colors.RED)
    else:
        typer.echo(f"\n✗ Unexpected error: {str(error)[:200]}", err=True, fg=typer.colors.RED)
    raise typer.Exit(1)


@app.command()
def generate(
    context: str = typer.Option(
        "birthday presents for a child",
        "--context", "-c",
        help="Context for evaluation criteria"
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output JSON file path (optional)"
    ),
    model: str = typer.Option(
        "gpt-4o",
        "--model", "-m",
        help="OpenAI model to use"
    ),
    temperature: float = typer.Option(
        0.7,
        "--temperature", "-t",
        help="Sampling temperature (0.0 to 2.0)",
        min=0.0,
        max=2.0
    )
):
    """
    Generate evaluation criteria for a given context.
    """
    typer.echo(f"Generating evaluation criteria for: {context}")
    
    try:
        criteria = generate_evaluation_criteria(
            context=context,
            model=model,
            temperature=temperature
        )
        
        typer.echo(f"\n✓ Generated {len(criteria.criteria)} criteria")
        typer.echo(f"Context: {criteria.context}")
        typer.echo(f"Total weight: {criteria.total_weight():.2f}")
        
        # Display criteria
        for i, criterion in enumerate(criteria.criteria, 1):
            typer.echo(f"\n{i}. {criterion.name} (weight: {criterion.weight})")
            typer.echo(f"   Description: {criterion.description}")
            if criterion.ideal_value:
                typer.echo(f"   Ideal: {criterion.ideal_value}")
        
        # Save to file if requested
        if output:
            with open(output, "w") as f:
                json.dump(criteria.model_dump(), f, indent=2)
            typer.echo(f"\n✓ Saved to {output}")
        
        # Show normalized weights
        typer.echo("\nNormalized weights (sum to 1.0):")
        normalized = criteria.normalized_weights()
        for criterion, weight in zip(criteria.criteria, normalized):
            typer.echo(f"  {criterion.name}: {weight:.3f}")
            
    except Exception as e:
        handle_error(e)


@app.command()
def chat(
    message: str = typer.Argument(..., help="Your message to the LLM"),
    model: str = typer.Option(
        "gpt-4o",
        "--model", "-m",
        help="OpenAI model to use"
    ),
    temperature: float = typer.Option(
        0.7,
        "--temperature", "-t",
        help="Sampling temperature (0.0 to 2.0)",
        min=0.0,
        max=2.0
    ),
    system_prompt: Optional[str] = typer.Option(
        None,
        "--system-prompt", "-s",
        help="System prompt for the LLM"
    )
):
    """
    Chat with the LLM about evaluation criteria.
    """
    try:
        response = chat_with_llm(
            user_message=message,
            model=model,
            temperature=temperature,
            system_prompt=system_prompt
        )
        typer.echo(response)
        
    except Exception as e:
        handle_error(e)


@app.command()
def validate(
    file: Path = typer.Argument(
        ...,
        help="JSON file containing evaluation criteria",
        exists=True
    )
):
    """
    Validate a JSON file against the EvaluationCriteria schema.
    """
    try:
        with open(file, "r") as f:
            data = json.load(f)
        
        # Validate by creating the model
        criteria = EvaluationCriteria(**data)
        typer.echo(f"✓ Valid EvaluationCriteria object")
        typer.echo(f"Context: {criteria.context}")
        typer.echo(f"Number of criteria: {len(criteria.criteria)}")
        
    except Exception as e:
        typer.echo(f"✗ Validation failed: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def converse(
    context: str = typer.Option(
        "", "--context", "-c", help="Initial context for the conversation"
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o", 
        help="Save successful criteria to JSON file"
    ),
    model: str = typer.Option(
        "gpt-4o-mini",
        "--model", "-m",
        help="Model to use for conversation"
    ),
    max_turns: int = typer.Option(
        10,
        "--max-turns", "-t",
        help="Maximum number of conversation turns",
        min=1,
        max=20
    )
):
    """
    Interactive conversation to generate evaluation criteria.
    
    Example:
    prompt-core converse --context "birthday presents for child"
    """
    typer.echo(f"Starting conversation using {model}... (Ctrl+C to quit, max {max_turns} turns)")
    
    try:
        orchestrator = ConversationOrchestrator(
            initial_context=context,
            max_turns=max_turns,
            model=model
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
            typer.echo(f"\n✗ Conversation ended without generating criteria")
            raise typer.Exit(1)
            
    except KeyboardInterrupt:
        typer.echo("\n\nConversation cancelled.")
        raise typer.Exit(0)
    except Exception as e:
        handle_error(e)


if __name__ == "__main__":
    app()