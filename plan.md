# Plan: Conversational Evaluation Criteria Generator

## Overview
Build a Python system where users converse with an LLM to generate structured `EvaluationCriteria` objects. The LLM acts as a facilitator, asking questions until it can produce valid criteria or gives up after reasonable effort.

## Core Requirements
1. **Interactive conversation** - LLM guides users through criteria generation
2. **Structured output** - Final result must be valid Pydantic `EvaluationCriteria` object
3. **Business rules**:
   - Must have at least 2 criteria
   - Must include a criterion named "budget" (case-insensitive)
4. **Three conversation states**:
   - `continue`: Ask another question
   - `success`: Return valid `EvaluationCriteria`
   - `failure`: Give up after reasonable effort
5. **Fail fast** - If LLM cannot return valid structured response after instructor's retries, fail immediately
6. **Turn limits** - Maximum 10 conversation turns (externally enforced)
7. **Full conversation history** - Include all messages in each LLM call, no truncation
8. **User-friendly errors** - Present clear error messages to users

## Architecture

### 1. Pydantic Models (Business Rules + Validation)

#### `Criterion` Model
```python
class Criterion(BaseModel):
    name: str = Field(..., description="Criterion name")
    description: str = Field(..., description="What this criterion measures")
    weight: float = Field(default=1.0, ge=0.0, le=10.0, description="Importance 0-10")
```

**Key points**:
- Use `Field()` with constraints: `ge=0.0, le=10.0` for weight
- Include examples: `Field(examples=["budget", "safety", "educational_value"])`
- No `is_budget_related` field - detect by name

#### `EvaluationCriteria` Model  
```python
class EvaluationCriteria(BaseModel):
    criteria: List[Criterion] = Field(default_factory=list)
    context: str = Field(default="General decision making")
    
    @model_validator(mode='after')
    def validate_business_rules(self):
        # Rule 1: Must have at least 2 criteria
        if len(self.criteria) < 2:
            raise ValueError("Must have at least 2 criteria")
        
        # Rule 2: Must include "budget" criterion (case-insensitive)
        if not any(c.name.lower() == "budget" for c in self.criteria):
            raise ValueError("Must include a criterion named 'budget' (case-insensitive)")
        return self
```

**Key points**:
- Use `@model_validator(mode='after')` for business rules
- Case-insensitive check: `name.lower() == "budget"`
- Include complete examples via `json_schema_extra` in `model_config`

### 2. Conversation Response Model

#### `ConversationAction` Model
```python
class ConversationAction(BaseModel):
    """LLM's decision about conversation flow with discriminator field."""
    action: Literal["continue", "success", "failure"]
    message: Optional[str] = None  # Required for "continue" and "failure" actions
    criteria: Optional[EvaluationCriteria] = None  # Required for "success" action
    
    @model_validator(mode='after')
    def validate_action_consistency(self):
        # Validate that the correct fields are provided based on action
        if self.action in ["continue", "failure"] and not self.message:
            raise ValueError(f"{self.action} action requires message")
        if self.action == "success" and not self.criteria:
            raise ValueError("success action requires criteria")
        return self
```

**Key Design**:
- Single response model with discriminator field (`action`)
- `action` determines which optional field is required
- Pydantic validation ensures consistency

### 3. Instructor Integration

#### How Instructor Works
- **Automatic schema generation**: `instructor.generate_openai_schema()` converts Pydantic models to OpenAI-compatible function schemas
- **Structured output**: `response_model` parameter ensures LLM returns validated Pydantic objects  
- **Automatic retries**: `max_retries` parameter handles validation failures (instructor retries up to N times)
- **No manual schema explanation needed**: Instructor handles all schema transformation

**Key points**:
- Use `response_model=ConversationAction` for all LLM calls
- Set `max_retries=3` to allow instructor to handle validation failures
- Instructor will automatically retry if LLM returns invalid structure

**Reference**: Instructor usage pattern in `/home/ashby/Documents/code/prompt-core-2.0t of the app

### 4. Conversation Orchestrator

#### `ConversationOrchestrator` Class Responsibilities:
1. **Manage conversation state** - Full message history, turn counting
2. **Enforce turn limits** - Maximum 10 turns (external to LLM)
3. **Call LLM with instructor** - Use `response_model=ConversationAction`
4. **Handle errors** - Convert instructor failures to user-friendly messages
5. **Maintain conversation flow** - Append assistant messages to history

#### Constructor:
```python
def __init__(self, initial_context: str = "", max_turns: int = 10, model: str = "gpt-4o-mini"):
    self.messages = []
    self.turn_count = 0
    self.max_turns = max_turns
    self.model = model  # Configurable model
    
    # Minimal system prompt - instructor will handle schema explanation
    system_prompt = """
    You are a helpful assistant that guides users through the process of creating structured information.
    You will have a multi-turn conversation to gather the information.
    Ask one question at a time to avoid overwhelming the user.
    Be empathetic and helpful.
    """
```

#### Conversation Result Class:
```python
class ConversationResult(BaseModel):
    """Result of a conversation turn."""
    criteria: Optional[EvaluationCriteria] = None
    message: str  # Message to show user
    is_complete: bool  # True if conversation ended (success or failure)
    
    @classmethod
    def continuing(cls, message: str) -> "ConversationResult":
        return cls(criteria=None, message=message, is_complete=False)
    
    @classmethod
    def success(cls, criteria: EvaluationCriteria) -> "ConversationResult":
        return cls(criteria=criteria, message="Criteria generated successfully!", is_complete=True)
    
    @classmethod
    def failure(cls, message: str) -> "ConversationResult":
        return cls(criteria=None, message=f"Failed: {message}", is_complete=True)
```

#### Key Method: `process_turn`
```python
def process_turn(self, user_input: str) -> ConversationResult:
    """
    Process one conversation turn.
    Returns: ConversationResult object with outcome
    
    Logic:
    1. Check turn limit (max 10)
    2. Add user message to history
    3. Call LLM via instructor with max_retries=3
    4. Handle ConversationAction:
        - "continue": Return ConversationResult.continuing(message)
        - "success": Return ConversationResult.success(criteria)
        - "failure": Return ConversationResult.failure(message)
    5. Handle exceptions (fail fast with user-friendly error)
    """
```

#### Error Handling in `_call_llm`:
```python
def _call_llm(self) -> ConversationAction:
    """Call LLM with instructor, handle retries and errors."""
    client = get_client()  # From existing llm_interaction.py
    
    try:
        return client.chat.completions.create(
            model=self.model,  # Configurable model
            messages=self.messages,
            response_model=ConversationAction,
            max_retries=3  # Instructor handles validation retries
        )
    except Exception as e:
        # Instructor's retries exhausted or other error
        # Extract user-friendly message from exception
        error_msg = str(e)
        # Look for user-facing text in error message
        # If none found, use generic error
        raise ValueError(f"System error: {self._extract_user_friendly_error(error_msg)}")
```

### 5. Main Entry Point Function

#### `converse_to_generate_criteria`:
```python
def converse_to_generate_criteria(
    initial_context: str = "",
    max_turns: int = 10,
    model: str = "gpt-4o-mini"  # Default to cheaper model
) -> EvaluationCriteria:
    """
    Conversational criteria generation.
    Raises ValueError with user-friendly message on failure.
    
    Usage:
    try:
        criteria = converse_to_generate_criteria("birthday presents for child")
    except ValueError as e:
        print(f"Failed: {e}")
    """
    orchestrator = ConversationOrchestrator(initial_context, max_turns, model)
    
    # If no initial context, start with empty user input
    if not initial_context:
        result = orchestrator.process_turn("")
    else:
        result = orchestrator.process_turn("Let's begin.")
    
    # For CLI: Interactive loop here
    # For library: Return orchestrator for caller to manage
    
    if result.is_complete:
        if result.criteria:
            return result.criteria
        else:
            raise ValueError(result.message)
    
    # In library mode, return orchestrator for continued interaction
    return orchestrator
```

### 6. CLI Interface

#### New `converse` Command:
```bash
# Basic usage
python -m prompt_core.cli converse --context "birthday presents for child"

# Save successful result
python -m prompt_core.cli converse --context "hiring criteria" --output criteria.json
```

#### CLI Implementation:
```python
@app.command()
def converse(
    context: str = typer.Option("", "--context", "-c", help="Initial context"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save to file"),
    model: str = typer.Option("gpt-4o-mini", "--model", "-m", help="Model to use")
):
    """Interactive conversation to generate criteria."""
    typer.echo(f"Starting conversation using {model}... (Ctrl+C to quit, max 10 turns)")
    
    try:
        orchestrator = ConversationOrchestrator(context, model=model)
        
        # Interactive loop
        while True:
            # Get user input
            user_input = typer.prompt("You")
            
            # Process turn
            result = orchestrator.process_turn(user_input)
            
            # Show LLM response
            typer.echo(f"\nAssistant: {result.message}")
            
            if result.is_complete:
                if result.criteria:
                    typer.echo(f"\n✓ Generated {len(result.criteria.criteria)} criteria")
                    for i, c in enumerate(result.criteria.criteria, 1):
                        typer.echo(f"{i}. {c.name}: {c.description} (weight: {c.weight})")
                    
                    if output:
                        with open(output, "w") as f:
                            json.dump(result.criteria.model_dump(), f, indent=2)
                        typer.echo(f"Saved to {output}")
                break
                
    except ValueError as e:
        typer.echo(f"✗ {e}", err=True)
        raise typer.Exit(1)
    except KeyboardInterrupt:
        typer.echo("\n\nConversation cancelled.")
        raise typer.Exit(0)
```

### 7. Error Handling Strategy

| Error Case | Source | Handling | User Message |
|------------|--------|----------|--------------|
| Instructor validation fails | instructor | Catch exception | Extract user-facing text or "Could not process LLM response" |
| Turn limit reached | ConversationOrchestrator | Return failure | "Maximum conversation turns (10) reached" |
| LLM returns `failure` action | LLM decision | Return failure | Show LLM's `message` field |
| Business rule validation fails | Pydantic in instructor | Instructor retries, then fail | Pydantic error message |
| Network/API error | OpenAI client | Catch exception | "Connection error: [simplified details]" |
| User interrupts | KeyboardInterrupt | Catch and exit | "Conversation cancelled" |

**Error message extraction**:
```python
def _extract_user_friendly_error(self, error_msg: str) -> str:
    """Extract most user-facing text from error message."""
    # Look for Pydantic validation errors (from instructor retries)
    if "validation error" in error_msg.lower() or "ValueError" in error_msg:
        # Try to extract the specific error
        import re
        # Look for error messages after "ValueError"
        matches = re.findall(r"ValueError[^:]*:\s*(.*?)(?:\n|$)", error_msg)
        if matches:
            return matches[0]
        # Look for any error message in quotes
        matches = re.findall(r'["\'](.*?)["\']', error_msg)
        if matches:
            return matches[0]
    
    # Look for instructor/LLM provider errors
    if "maximum retries" in error_msg.lower() or "max_retries" in error_msg:
        return "Could not get valid response from AI after multiple attempts"
    
    # Look for API/connection errors
    if "api key" in error_msg.lower() or "authentication" in error_msg.lower():
        return "API authentication error - check your API key"
    
    if "connection" in error_msg.lower() or "timeout" in error_msg.lower():
        return "Connection error - check your network"
    
    # Generic fallback - show first 100 chars
    return f"Error: {error_msg[:100]}..."
```

### 8. File Structure Changes

```
prompt-core/
├── prompt_core/
│   ├── __init__.py              # Export new classes
│   ├── models.py                # Enhanced with business rules
│   ├── conversation.py          # NEW: ConversationAction, ConversationOrchestrator
│   ├── llm_interaction.py       # Updated with converse_to_generate_criteria()
│   └── cli.py                   # Updated with converse command
├── example.py                   # Updated with conversation example
├── test_conversation.py         # NEW: Tests for conversation flow
├── plan.md                      # THIS FILE
└── pyproject.toml              # Dependencies already satisfied
```

### 9. Implementation Steps

#### Phase 1: Core Models & Validation (1-2 hours)
1. Update `models.py`:
   - Add `@model_validator` for business rules
   - Add examples with proper "budget" criterion
   - Ensure case-insensitive budget check
   - Add `json_schema_extra` with complete examples

#### Phase 2: Conversation Response Model (1 hour)
1. Create `conversation.py`:
   - Define `ConversationAction` with discriminator field
   - Implement validation for action-field consistency

#### Phase 3: Conversation Orchestrator (2 hours)
1. Complete `conversation.py`:
   - Define `ConversationResult` class (proper return type)
   - Implement `ConversationOrchestrator` class
   - Add minimal system prompt (no validation duplication)
   - Implement turn limiting
   - Add error message extraction
   - Implement `_call_llm` method with configurable model

#### Phase 4: Integration (1 hour)
1. Update `llm_interaction.py`:
   - Add `converse_to_generate_criteria()` function
   - Reuse existing `get_client()` function
   - Import `ConversationOrchestrator`

#### Phase 5: CLI Interface (1 hour)
1. Update `cli.py`:
   - Add `converse` command with Typer
   - Add `--model` parameter for model selection
   - Implement interactive loop using `ConversationResult`
   - Add save option for successful results

#### Phase 6: Testing (1 hour)
1. Create `test_conversation.py`:
   - Mock tests for conversation flow
   - Test business rule validation
   - Test error conditions
   - Test turn limiting

### 9. Key Technical Decisions

#### 1. **Action-based Response Model**
- **Why**: Clear discriminator field (`action`) makes it easy for LLM to understand
- **Instructor compatibility**: Single model is easier for instructor to handle than Union types
- **Pydantic-only validation**: All business rules encoded in Pydantic, not duplicated in prompts
- **Minimal prompts**: Instructor handles schema explanation; prompts only guide conversation style
- **Model-agnostic**: Support any LLM provider via instructor, default to cheaper models
- **Full Conversation History**: LLM needs full context; 10-turn limit mitigates token concerns
- **External Turn Limiting**: Prevents infinite loops without LLM gaming the system
- **Case-Insensitive "budget"**: Accepts "Budget", "BUDGET", etc. while maintaining requirement
- **Fail Fast**: Better user experience than retrying indefinitely
- **Proper return types**: Use `ConversationResult` class instead of tuple

### 10. Implementation Steps

#### Phase 1: Core Models & Validation (1 hour)
1. Update `models.py`:
   - Add `@model_validator` for business rules
   - Add examples with proper "budget" criterion
   - Ensure case-insensitive budget check: `name.lower() == "budget"`

#### Phase 2: Conversation Response Model (1 hour)
1. Create `conversation.py`:
   - Define `ConversationAction` with discriminator field
   - Implement `validate_action_consistency` method

#### Phase 3: Conversation Orchestrator (2 hours)
1. Complete `conversation.py`:
   - Implement `ConversationOrchestrator.__init__` with system prompt
   - Implement `process_turn` method
   - Add `_call_llm` with error handling
   - Implement `_extract_user_friendly_error` helper

#### Phase 4: Integration (1 hour)
1. Update `llm_interaction.py`:
   - Add `converse_to_generate_criteria()` function
   - Import `ConversationOrchestrator`
   - Add user-friendly error wrapping

#### Phase 5: CLI Interface (1 hour)
1. Update `cli.py`:
   - Add `converse` command with Typer
   - Implement interactive loop with prompts
   - Add save option for successful results
   - Handle `KeyboardInterrupt`

#### Phase 6: Testing & Examples (1 hour)
1. Create `test_conversation.py`:
   - Mock tests for conversation flow
   - Test business rule validation
   - Test error conditions
2. Update `example.py`:
   - Add conversation example
   - Show usage patterns

### 11. Testing Strategy

#### Unit Tests:
1. **Business rule validation**:
   - Test `EvaluationCriteria` with 1 criterion (should fail)
   - Test without "budget" criterion (should fail)
   - Test with "Budget" (uppercase, should pass)
   - Test with "budget" (lowercase, should pass)

2. **ConversationAction validation**:
   - Test `action="continue"` without `message` (should fail)
   - Test `action="success"` without `criteria` (should fail)
   - Test `action="failure"` without `message` (should fail)
   - Test valid combinations (should pass)

3. **Turn limiting**:
   - Test orchestrator stops after max turns
   - Test turn counting logic

#### Integration Tests (with mocks):
1. **Happy path**: Mock LLM returns `success` with valid criteria
2. **Conversation path**: Mock LLM returns `continue` then `success`
3. **Failure path**: Mock LLM returns `failure` after questions
4. **Error path**: Mock instructor to raise exceptions

#### Manual Testing Checklist:
- [ ] Empty context starts conversation appropriately
- [ ] LLM asks relevant questions about child, budget, etc.
- [ ] Valid criteria generation with "budget" criterion
- [ ] Case-insensitive "budget" detection works
- [ ] Turn limit enforced after 10 turns
- [ ] User-friendly error messages for all error cases
- [ ] CLI save functionality works
- [ ] Ctrl+C cancels conversation cleanly

### 12. Dependencies
Already satisfied in `pyproject.toml`:
- `instructor>=1.15.1`: Structured LLM outputs, schema generation, retries
- `pydantic>=2.13.3`: Data validation, business rules
- `openai>=2.32.0`: LLM API access
- `typer>=0.24.1`: CLI interface
- `python-dotenv>=1.2.2`: Environment configuration

### 13. Reference Files
For implementation patterns, refer to:
- **Business rules & examples**: `/home/ashby/Documents/code/prompt-core/complete_example.py` (lines 58-136)
- **Instructor usage**: `/home/ashby/Documents/code/prompt-core/prompt_core/llm_interaction.py` (lines 59-68) - Note: Update `get_client()` to support multiple providers
- **Research findings**: `/home/ashby/Documents/code/prompt-core/RESEARCH_FINDINGS.md` (covers Pydantic best practices)

### 14. Success Criteria
1. **Functional**: User can converse with LLM and get valid `EvaluationCriteria`
2. **Validation**: Business rules enforced (≥2 criteria, includes "budget")
3. **Error handling**: All errors result in clear user-friendly messages
4. **Usability**: CLI is intuitive, conversation flows naturally
5. **Robustness**: Handles malformed LLM responses via instructor retries
6. **Performance**: 10-turn limit prevents excessive costs/loops

### 15. Open Questions & Decisions
1. **Error message extraction**: Need to examine actual exception structure from instructor to extract user-facing text
2. **LLM provider support**: Instructor supports multiple providers (OpenAI, Anthropic, etc.) - need to configure `get_client()` appropriately
3. **Model selection**: Default to `gpt-4o-mini` for cost savings, but allow override
4. **Testing mocks**: Need to mock LLM client for tests without API calls
5. **Instructor provider modes**: May need to configure instructor mode based on provider

This plan provides complete implementation guidance. The system leverages instructor's capabilities fully and follows Pydantic best practices for LLM applications.
