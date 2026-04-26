# Quick Start for New Contributors

## 30-Second Overview
- **Goal**: Generate evaluation criteria via LLM conversation
- **Core**: `ConversationOrchestrator` manages multi-turn dialogue  
- **Output**: `EvaluationCriteria` Pydantic model (validated)
- **Key rule**: Must include "budget" criterion, ≥2 total criteria

## First 5 Files to Read
1. `prompt_core/models.py` - Data structures & validation rules
2. `prompt_core/conversation.py` - Conversation logic & prompts
3. `tests/unit/test_models.py` - See business rule tests
4. `tests/unit/test_orchestrator_logic.py` - See conversation flow tests
5. `spec.md` - Product requirements

## Critical Code Locations
```python
# Business rules (MUST maintain):
prompt_core/models.py:33  # validate_business_rules()

# Conversation flow:
prompt_core/conversation.py:53  # System prompt template
prompt_core/conversation.py:82  # process_turn() - main logic

# LLM integration:
prompt_core/llm_interaction_multi.py:97  # get_client() - provider setup
```

## Prompt Design Rules
1. **NEVER** mention Pydantic class names in prompts
2. **DO** give behavioral examples & conversation strategies  
3. **ALWAYS** make turn limit configurable via f-string
4. **FOCUS** on "what to do" not "what not to do"

## Configuration
```bash
# 1. Copy config template
# Edit config.json to set provider/model

# 2. Edit config.json to set provider/model
# 3. Set API key environment variable matching provider
export OPENROUTER_API_KEY=your-key-here  # if provider is "openrouter"
```

## Common Commands
```bash
# Run all unit tests (mocks only)
make test

# Check prompt changes work
python -c "from prompt_core.conversation import ConversationOrchestrator; o=ConversationOrchestrator(max_turns=5); print(o.messages[0]['content'][:200])"

# Check configuration
python -c "from prompt_core.config import config; print(config.provider, config.model)"
```

## When You're Stuck
1. **Business logic issue?** → Check `models.py` validation
2. **Conversation flow problem?** → Check `conversation.py` prompts & logic
3. **LLM integration failing?** → Check `llm_interaction_multi.py`
4. **Test failing?** → Check if it's a real API test needing keys