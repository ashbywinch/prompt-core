# Testing Strategy

This document describes the testing approach for the conversation orchestration system.

## Test Philosophy

1. **Fail fast with exceptions**: Code raises exceptions instead of returning failure objects
2. **Test separation**: Clear separation between unit tests (mocked) and integration tests (real API)
3. **Infrastructure exposure**: Tests that use real API **FAIL (not skip)** when API key missing
4. **Proper mocking**: Test logic with mocks, test prompts with real API
5. **Intentional failures**: Missing API keys cause test failures to expose missing infrastructure

**IMPORTANT**: If tests fail due to missing API keys, the correct fix is to **configure API keys**, not to modify tests to skip or mock.

## Test Structure

```
tests/
├── unit/                     # Unit tests (mocked)
│   ├── test_models.py       # Pydantic models (no mocking)
│   ├── test_orchestrator_logic.py  # Orchestrator logic (mock _call_llm)
│   └── test_llm_interaction.py     # LLM interaction (mock get_client)
├── integration/              # Integration tests
│   └── test_real_api.py     # Real API tests (require OPENAI_API_KEY)
└── conftest.py              # Common test configuration
```

## Key Test Patterns

### 1. Orchestrator Logic Tests (mock `_call_llm`)
- Test conversation flow without LLM dependencies
- Mock sequence of responses to test multi-turn conversations
- Test exception handling for turn limits and invalid actions

**Example:**
```python
@patch.object(ConversationOrchestrator, '_call_llm')
def test_multi_turn_conversation(self, mock_call_llm):
    """Test orchestrator handles sequence of turns correctly."""
    orchestrator = ConversationOrchestrator(max_turns=5)
    
    # Set up sequence of mock responses
    responses = [
        ConversationAction(action="continue", message="Question 1"),
        ConversationAction(action="continue", message="Question 2"),
        ConversationAction(action="success", criteria=valid_criteria)
    ]
    mock_call_llm.side_effect = responses
    
    # Test each turn
    result1 = orchestrator.process_turn("Hello")
    assert not result1.is_complete
    assert result1.message == "Question 1"
```

### 2. LLM Interaction Tests (mock `get_client`)
- Test `_call_llm()` method with mocked OpenAI client
- Test error handling for API failures and validation errors
- Verify correct parameters passed to instructor

**Example:**
```python
@patch('prompt_core.llm_interaction.get_client')
def test_call_llm_success(self, mock_get_client):
    """Test successful LLM call returns ConversationAction."""
    mock_client = Mock()
    expected_action = ConversationAction(action="continue", message="Test")
    mock_client.chat.completions.create.return_value = expected_action
    mock_get_client.return_value = mock_client
    
    action = self.orchestrator._call_llm()
    assert action == expected_action
```

### 3. Real API Tests (no mocking)
- Test actual conversation with real LLM
- **FAIL** (not skip) when `OPENAI_API_KEY` missing
- Provide clear error messages about missing infrastructure
- **Verification goal**: Ensure our prompts generate valid structured responses from real LLMs

**Purpose**: These tests verify that:
1. Our prompts work correctly with real LLMs
2. LLMs generate properly formatted `ConversationAction` responses
3. Business rules (minimum 2 criteria, must include "budget") are enforced

**Expectation**: These tests will **fail** without API key. This is **intentional** - it exposes missing infrastructure.

## API Key Requirements

### Tests That Require API Keys

The following tests require valid API keys and will fail if not configured:

1. **Unit tests** (in `tests/unit/test_llm_interaction.py`):
   - `test_call_llm_success` - Tests single turn with real LLM
   - `test_multi_turn_conversation_with_real_llm` - Tests conversation flow

2. **Integration tests** (in `tests/integration/test_real_api.py`):
   - All tests in this file require API keys

### Configuring API Keys

**Step 1: Choose a provider**
- **Google Gemini** (recommended for free tier): [Get API key](https://makersuite.google.com/app/apikey)
- **OpenAI**: [Get API key](https://platform.openai.com/api-keys) - $5 free credits
- **Groq**: [Get API key](https://console.groq.com) - Free tier available

**Step 2: Configure `.env` file**
```bash
# Copy example file
cp .env.example .env

# Edit .env and add your API key
# For Google Gemini:
GOOGLE_API_KEY=AIza...

# For OpenAI:
OPENAI_API_KEY=sk-...

# Optional: Specify provider (defaults to "openai")
LLM_PROVIDER=google
```

**Step 3: Verify configuration**
```bash
# Test that API key is accessible
python -c "import os; print('OPENAI_API_KEY:', 'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET')"

# Run a real API test (should pass with key, fail without)
python -m unittest tests.unit.test_llm_interaction.TestLLMInteraction.test_call_llm_success
```

### Troubleshooting Test Failures

| Error Message | What it means | Solution |
|---------------|---------------|----------|
| `ValueError: OpenAI API key not provided` | No API key configured | Add API key to `.env` file |
| `ValueError: Provider 'openai' not available` | Provider package not installed | Install with `pip install openai` |
| Tests pass without API key | Tests are only using mocks | This is normal for mocked unit tests |
| All real API tests fail | API key may be invalid | Verify API key is correct and has credits |

### Philosophy: Why Tests Fail Without API Keys

1. **Infrastructure testing**: Tests verify the entire stack works, including API connectivity
2. **Prompt validation**: Real LLM responses validate that our prompts generate correct structured data
3. **Early failure**: Better to fail in tests than in production
4. **Documentation**: Test failures document what infrastructure is required

**DO NOT** modify tests to skip or mock API calls. Instead, **configure the required API keys**.

### Running Different Test Sets

```bash
# Run ALL tests (some will fail without API key)
make test-all

# Run only mocked unit tests (should pass without API key)
make test-unit

# Run real API tests (REQUIRES API KEY)
make test-integration

# Run specific test that requires API key
python -m unittest tests.unit.test_llm_interaction.TestLLMInteraction.test_call_llm_success -v

### Diagnostic Tool

If tests fail due to missing API keys, run the diagnostic script:

```bash
python scripts/check_api_keys.py
```

This will:
1. Check which API keys are configured
2. Verify `.env` file setup
3. Suggest appropriate actions
4. Provide links to get free API keys
```

**Important**: These tests make real API calls and may incur costs.

## Running Tests

### Using Makefile (recommended)
```bash
make help           # Show all commands
make test           # Run all unit tests
make test-unit      # Run all unit tests under tests/unit/
make test-basic     # Run basic model tests only (no mocking)
make test-integration  # Run integration tests (requires OPENAI_API_KEY)
make test-all       # Run ALL tests (unit + integration)
make coverage       # Run tests with coverage report
make lint           # Run code linting
make clean          # Clean up generated files
```

### Direct Python Commands
```bash
# Unit tests
python test_conversation.py    # All unit tests
python test_basic.py           # Basic model tests only

# Integration tests (will fail without API key)
python -m unittest discover tests/integration/
```

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          python -m venv .venv
          source .venv/bin/activate
          pip install -r requirements.txt
      - name: Run unit tests
        run: |
          source .venv/bin/activate
          make test
      - name: Run integration tests (if API key available)
        if: env.OPENAI_API_KEY != ''
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          source .venv/bin/activate
          make test-integration
```

## Test Coverage Goals

- **Models**: 100% (validation, business rules)
- **Orchestrator logic**: 100% (turn management, action handling)
- **LLM interaction**: 100% (retry logic, error handling)
- **Integration**: Critical paths only (real API interaction)

## Error Handling in Tests

Tests verify that:
1. Max turn limit raises `ValueError`
2. LLM failure action raises `ValueError` 
3. API errors propagate as exceptions
4. Invalid actions raise `ValueError`

**No error extraction in orchestrator**: Error formatting is a CLI concern, not orchestrator responsibility.