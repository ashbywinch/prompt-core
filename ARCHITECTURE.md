# Prompt Core - Architecture Overview

## Quick Navigation Guide

### **Core Files (Read These First)**
1. `prompt_core/models.py` - Data models & business rule validation
2. `prompt_core/conversation.py` - Conversation orchestration & prompts  
3. `prompt_core/llm_interaction_multi.py` - LLM provider abstraction

### **Configuration Files**
- `config.json` - LLM provider and model settings (REQUIRED)
- Environment variables - API keys (e.g., `OPENAI_API_KEY`, `OPENROUTER_API_KEY`)
- `config.py` - Loads and manages configuration from `config.json`

### **Key Concepts**
- **Structured outputs via Instructor**: LLM returns Pydantic objects, not JSON
- **Validation-first**: Business rules in Pydantic models, not prompts
- **Multi-turn conversation**: Stateful orchestrator manages dialogue flow
- **Configurable failure modes**: LLM can fail OR system can hit turn limits
- **Dual configuration**: Provider/model in `config.json`, API keys in environment

## File Responsibilities

### `models.py` - Data Validation
```python
# Business rules enforced here:
EvaluationCriteria.validate_business_rules()
# - Must have ≥2 criteria
# - Must include "budget" criterion (case-insensitive)
```

### `conversation.py` - Conversation Logic
```python
# Core class: ConversationOrchestrator
# - Manages turn state (max_turns configurable)
# - Contains system prompt (behavioral guidance only)
# - Three outcomes: continue/success/failure
# - Prompt avoids Pydantic schema duplication
```

### `llm_interaction_multi.py` - LLM Abstraction
```python
# Unified client for multiple providers
get_client()  # Returns instructor-patched client
# Supports: OpenAI, Google, OpenRouter, etc.
```

### `config.py` - Configuration Management
```python
# Singleton configuration manager
config = Config()  # Reads ONLY from config.json
# Provides: provider, model, temperature, max_retries
# Note: API keys come from environment variables, not config.json
```

### `llm_provider.py` - Provider Interface
```python
# Abstract base class for LLM providers
LLMProvider  # Defines interface for structured responses
get_provider()  # Factory function to get provider instance
```

## Critical Patterns

### 1. Prompt Design Philosophy
- **DO**: Give behavioral guidance, examples, conversation strategy
- **DON'T**: Mention Pydantic class names or schema details (Instructor handles this)
- **Example**: Say "ask about budget constraints" not "create a 'budget' criterion"

### 2. Validation Layers
```
User Input → LLM Response → Instructor → Pydantic Validation
                                  ↓
                    Business Rules Enforced
```

### 3. Failure Modes
- **LLM-initiated**: Returns `action: "failure"` for unconstructive users
- **System-initiated**: `ValueError` when `max_turns` exceeded
- **Validation failure**: Pydantic raises error for invalid criteria

## Common Tasks & Where to Look

| Task | Primary File | Key Function/Method |
|------|--------------|---------------------|
| Add business rule | `models.py` | `EvaluationCriteria.validate_business_rules()` |
| Modify prompt | `conversation.py` | `ConversationOrchestrator.__init__()` |
| Change LLM provider | `llm_interaction_multi.py` | `get_client()` |
| Add conversation outcome | `conversation.py` | `ConversationAction` model |
| Add test for new feature | `tests/unit/` | Follow existing test patterns |

## Testing Strategy
- **Unit tests**: `tests/unit/` - Mock LLM, test logic
- **Integration tests**: `tests/integration/` - Real API calls (require keys)
- **Key principle**: Tests fail without API keys (exposes setup issues)

## Quick Start for Common Changes

### Change Business Rules
1. Edit `models.py` `validate_business_rules()`
2. Update corresponding tests in `tests/unit/test_models.py`

### Modify Conversation Flow  
1. Edit `conversation.py` system prompt (focus on behavior)
2. Check `ConversationOrchestrator.process_turn()` logic
3. Update `tests/unit/test_orchestrator_logic.py`

### Add LLM Provider
1. Update `llm_interaction_multi.py` `get_client()`
2. Add provider configuration handling

## Important Notes
- Turn limit appears in prompt AND code (both need updating if changed)
- Prompts should never duplicate Instructor's schema explanations
- All validation errors should be user-friendly
- API keys are optional for unit tests, required for integration tests