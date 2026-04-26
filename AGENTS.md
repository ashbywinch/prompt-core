# AGENTS.md — Prompt Core

AI agent onboarding. For human onboarding, see [README.md](README.md).

## What This Project Does

Generates structured `EvaluationCriteria` objects via multi-turn LLM conversation. The LLM guides a user through questions, then produces a validated Pydantic model with weighted criteria (min 2, must include "budget").

## Key Files

| File | What |
|------|------|
| `prompt_core/models.py` | Data models & business rules (`EvaluationCriteria`, `Criterion`) |
| `prompt_core/conversation.py` | `ConversationOrchestrator`, `ConversationAction` models, system prompt |
| `prompt_core/llm_interaction.py` | `get_client()` — multi-provider LLM client via instructor+litellm |
| `prompt_core/config.py` | Singleton `Config()` — reads `config.json` for provider/model/timeout |
| `prompt_core/exceptions.py` | Custom exception hierarchy |
| `prompt_core/cli.py` | Typer CLI (`converse` command) |

## Conventions

- **Business rules live in Pydantic models** (`model_validator`), not prompts.
- **Prompts give behavioral guidance only** — Instructor handles schema formatting.
- **`max_turns`** appears in both the system prompt (f-string) and the code guard.
- **Tests fail (not skip) without API keys** — this exposes missing infrastructure intentionally.
- **Custom exceptions** for all error cases — CLI formats them, orchestrator raises them.

## Commands

```bash
make test          # Unit tests only (no API key needed, ~0.01s)
make test-verbose  # Same with verbose output per test
make evals          # Real-API evals (requires config.json + API key, ~90s)
make evals-verbose  # Same with verbose output
make lint           # black --check + ruff check
```

## Prompt Design Rules

1. NEVER mention Pydantic class names, field types or validation rules in prompts.
2. DO give behavioral examples and conversation strategies.
3. ALWAYS use f-strings for turn limits (`{self.max_turns}`), never hardcode.
4. FOCUS on "what to do", not "what not to do".

## Critical Patterns

- All development must be done on a branch. origin/main is protected.
- `ConversationAction` is a single discriminator model with `action: Literal["continue", "success", "failure"]`
- `ConversationOrchestrator.process_turn()` checks turn limit, calls LLM, handles action
- Turn limit raises `TurnLimitExceededError`; failure action raises `ConversationFailedError`

## Reference Docs

- [ARCHITECTURE.md](ARCHITECTURE.md) — Deeper architecture and file responsibilities
- [QUICKSTART.md](QUICKSTART.md) — 5-minute contributor guide with critical code locations
- [spec.md](spec.md) — Product specification and philosophy
- [docs/TESTING.md](docs/TESTING.md) — Full testing strategy and patterns