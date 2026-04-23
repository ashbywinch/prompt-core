# Prompt Core

A Python library for generating structured evaluation criteria using LLMs with instructor and Pydantic.

## Specification 

See spec.md 

## Installation

### Using uv (recommended)

```bash
# Clone the repository
git clone <repository-url>
cd prompt-core

# Install dependencies
make

```

## Quick Start

### 1. Set up configuration

#### Step 1: Configure LLM provider and model
Copy `config.json.example` to `config.json` and edit:

```bash
cp config.json.example config.json
```

Edit `config.json` to set your preferred LLM provider and model:
```json
{
  "llm": {
    "provider": "openrouter",  # or "openai", "google", "groq", "anthropic"
    "model": "openrouter/xiaomi/mimo-v2-flash",
    "temperature": 0.7,
    "max_retries": 3
  }
}
```

#### Step 2: Set API key environment variable
Based on your provider from Step 1, set the corresponding API key:

```bash
# For OpenRouter (recommended for testing - no rate limits)
export OPENROUTER_API_KEY=your-openrouter-api-key-here

# For OpenAI
export OPENAI_API_KEY=your-openai-api-key-here

# For Google Gemini (most generous free tier)
export GOOGLE_API_KEY=your-google-api-key-here

# For Groq (fast, free tier available)
export GROQ_API_KEY=your-groq-api-key-here
```

**Note**: You need both `config.json` AND the corresponding API key environment variable.

### 2. Run the example

```bash
python example.py
```

### 3. Use the CLI

Generate evaluation criteria:

```bash
# Basic usage
python -m prompt_core.cli generate --context "birthday presents for a 7-year-old"

# Save to file
python -m prompt_core.cli generate --context "hiring a software engineer" --output criteria.json

# Custom model and temperature
python -m prompt_core.cli generate --context "choosing a vacation destination" --model gpt-4-turbo --temperature 0.5
```

### 3. Use the CLI

```bash
# Generate criteria interactively
prompt-core converse "birthday presents for a 7-year-old"

# Generate criteria directly
prompt-core generate "evaluating coffee makers"
```

## Testing

### Test Philosophy

Tests are designed to **fail when infrastructure is missing**, not skip silently. This exposes missing dependencies early.

**Key principle**: If tests fail due to missing API keys, the fix is to **configure API keys**, not to mock or skip tests.

### Running Tests

```bash
# Run all tests (some will FAIL without API key)
make test

# Run only unit tests with mocks (should pass without API key)
make test-unit

# Run integration tests (REQUIRES API KEY - will FAIL without)
make test-integration

# Run tests that verify prompts work with real LLM (REQUIRES API KEY)
python -m unittest tests.unit.test_llm_interaction.TestLLMInteraction.test_call_llm_success

# Run specific real API tests
python tests/integration/test_real_api.py
```

### Test Failures and Solutions

| Test Failure | What it means | How to fix |
|--------------|---------------|------------|
| `ValueError: OpenAI API key not provided` | Tests are trying to use real LLM API | Set API key environment variable |
| Any test mentioning "API key" or "authentication" | Missing API configuration | Configure both `config.json` AND API key environment variable |
| Tests pass without API key | Tests are only using mocks | No action needed - but consider running real API tests |

### Getting API Keys

**Recommended free options for testing (avoids rate limits):**
1. **OpenRouter**: [Get API key](https://openrouter.ai/keys) - Free router with no rate limits, ideal for testing
2. **Google Gemini**: [Get API key](https://makersuite.google.com/app/apikey) - Most generous free tier
3. **Groq**: [Get API key](https://console.groq.com) - Fast, free tier available  
4. **OpenAI**: [Get API key](https://platform.openai.com/api-keys) - $5 free credits for new users

**Required setup:**
1. **Create `config.json`** from `config.json.example`
2. **Set API key environment variable** matching your provider choice in `config.json`

Example for OpenRouter:
```bash
cp config.json.example config.json
# Edit config.json to set provider: "openrouter"
export OPENROUTER_API_KEY=your-openrouter-api-key
```

See [TESTING.md](TESTING.md) for detailed testing strategy.

## Troubleshooting

### Tests Fail with "API key not provided"

This is **expected behavior** when API keys are not configured. The tests are designed to fail when infrastructure is missing.

**Quick diagnostic:**
```bash
python scripts/check_api_keys.py
```

**Solutions:**
1. **Get a free API key** (recommended: Google Gemini for free tier)
2. **Create `config.json`**: `cp config.json.example config.json` and edit provider/model
3. **Set API key environment variable** matching your provider
4. **Verify setup**: Run `make test-integration` (should pass with API key)

### Tests Pass Without API Key

Some unit tests use mocks and will pass without API keys. This is normal. Only tests that verify real LLM interaction require API keys.

### Getting Help

- Check [TESTING.md](TESTING.md) for detailed troubleshooting
- Run diagnostic script: `python scripts/check_api_keys.py`
- Review test output for specific error messages

Validate JSON files:

```bash
python -m prompt_core.cli validate criteria.json
```

## Python API

### Generate evaluation criteria

```python
from prompt_core import generate_evaluation_criteria

criteria = generate_evaluation_criteria(
    context="birthday presents for a 7-year-old child who loves science",
    model="gpt-4o",
    temperature=0.7
)

print(f"Generated {len(criteria.criteria)} criteria")
print(f"Context: {criteria.context}")

for criterion in criteria.criteria:
    print(f"- {criterion.name}: {criterion.description} (weight: {criterion.weight})")
```

### Chat with LLM

```python
from prompt_core import chat_with_llm

response = chat_with_llm(
    "Based on these criteria for birthday presents, suggest 3 gift ideas.",
    system_prompt="You are a helpful gift advisor."
)
print(response)
```

### Work with the model directly

```python
from prompt_core import EvaluationCriteria, Criterion

# Create criteria manually
criteria = EvaluationCriteria(context="evaluating job offers")
criteria.add_criterion(
    name="Salary",
    description="Annual compensation including bonuses",
    weight=8.5,
    ideal_value="Above market average"
)
criteria.add_criterion(
    name="Work-life balance",
    description="Flexibility and reasonable hours",
    weight=9.0
)

# Calculate normalized weights
normalized = criteria.normalized_weights()
print(f"Salary importance: {normalized[0]:.1%}")
```

## Project Structure

```
prompt-core/
├── prompt_core/
│   ├── __init__.py          # Exports
│   ├── models.py            # Pydantic models (EvaluationCriteria, Criterion)
│   ├── llm_interaction.py   # LLM functions
│   └── cli.py              # CLI interface
├── example.py              # Example usage
├── pyproject.toml         # Project dependencies
├── .env.example           # Environment template
└── README.md             # This file
```

## Models

### `Criterion`
- `name`: Name of the criterion
- `description`: Detailed description
- `weight`: Importance weight (0.0 to 10.0)
- `ideal_value`: Optional target value

### `EvaluationCriteria`
- `criteria`: List of `Criterion` objects
- `context`: Description of the evaluation context
- Methods: `add_criterion()`, `total_weight()`, `normalized_weights()`

## Use Cases

1. **Gift selection**: Evaluate birthday/Christmas gift options
2. **Hiring decisions**: Criteria for evaluating job candidates
3. **Product selection**: Comparing products or services
4. **Project prioritization**: Deciding which projects to pursue
5. **Personal decisions**: Making complex life choices

## Extending

The project is designed to be extended:

1. Add new fields to `Criterion` model
2. Create specialized criteria generators for different domains
3. Integrate with other LLM providers
4. Add scoring functions to evaluate options against criteria
5. Create web interface or GUI

## Architecture & Development

For developers and AI agents working with the codebase:

### Quick Navigation
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Comprehensive architecture overview
- **[QUICKSTART.md](QUICKSTART.md)** - 5-minute guide for new contributors  
- **[.agent-navigation](.agent-navigation)** - Structured hints for AI agents

### Key Files to Understand First
1. `prompt_core/models.py` - Data models & business rule validation
2. `prompt_core/conversation.py` - Conversation orchestration & prompts
3. `prompt_core/llm_interaction_multi.py` - LLM provider abstraction

### Critical Patterns
- Business rules enforced in Pydantic models (not prompts)
- Prompts focus on behavioral guidance (not schema details)
- Turn limits configurable and appear in both code and prompts
- Tests fail without API keys to expose infrastructure issues

## License

MIT
