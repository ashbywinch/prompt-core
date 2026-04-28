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

## Git Workflow

**After making a PR or at the start of a new session**

1. Decide whether there is outstanding work on the current branch. This may include changes that are not committed, not pushed, not part of a PR, or part of a PR that has not been merged yet, perhaps because it doesn't build or there is some problem with it.

```bash
# Get PR status and branch name
gh pr view --json state,headRefName,number --jq '[.state, .headRefName, .number] | @tsv'
gh pr checks <number>
```

2. If there is outstanding work, encourage the user to complete that work first before continuing.

3. Only once the existing work is fully merged to main, start the new work:

```bash
# Start new work (only after previous PR is merged)
git branch -d <last-branch>                 # Safe: -d refuses if unmerged
git checkout main && git pull origin main && git checkout -b <new-branch>
```
If branch -d fails because there are outstanding changes then return to step 2

**To push changes**

1. Run ALL tests locally before pushing:
   ```bash
   make test    # Unit tests, fast
   make evals   # Real API tests, requires API key, ~90s
   ```

2. If tests/evals fail locally: fix them before pushing. CI will run them again and fail the same way.

**When making a PR or pushing to an existing PR**

After pushing, verify CI passes:
   ```bash
   gh pr checks <number> --watch
   ```

**Never:**
- Push changes that break tests (unit or eval) without fixing them first
- Assume unit tests passing means everything works — evals test real LLM behavior
- Merge a PR that has failing CI

**Rules:**
- Start every new piece of work from a fresh branch off main. Never reuse a branch whose PR has been merged.
- If your PR is open but not yet merged: wait, or ask. Don't push more commits without confirmation.

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
