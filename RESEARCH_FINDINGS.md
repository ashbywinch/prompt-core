# Research Findings: Pydantic Best Practices for LLM Applications

Based on my research of the codebase and Python ecosystem, here are the best practices for your four questions:

## 1. **Encoding Business Rules in Pydantic Models**

### **Recommended Approaches:**

**A. Field-level Constraints** (for simple rules):
```python
weight: float = Field(default=1.0, ge=0.0, le=10.0)  # Must be 0-10
name: str = Field(..., min_length=1, max_length=100)  # Length constraints
```

**B. Model Validators** (for complex business rules - **RECOMMENDED**):
```python
@model_validator(mode='after')
def validate_business_rules(self):
    # Rule 1: Must have >1 criteria
    if len(self.criteria) < 2:
        raise ValueError('Must have at least 2 criteria')
    
    # Rule 2: Must include budget criterion if required
    if self.require_budget and not any(c.is_budget for c in self.criteria):
        raise ValueError('Must include budget criterion')
    
    # Rule 3: Total weight constraints
    total = sum(c.weight for c in self.criteria)
    if not 5.0 <= total <= 50.0:
        raise ValueError(f'Total weight {total} outside range 5-50')
    
    return self
```

**C. Custom Types** (for reusable validation):
```python
from typing import Annotated
from pydantic import AfterValidator

Percentage = Annotated[float, AfterValidator(lambda x: x if 0 <= x <= 100 else ValueError('0-100 only'))]
```

**Key Insights:**
- Use `@model_validator(mode='after')` for cross-field validation
- Field validators (`@field_validator`) for single-field validation with dependencies
- Business rules should raise clear `ValueError` messages
- Consider `@property` methods for computed validations that don't block instantiation

## 2. **Explaining Pydantic Schemas to LLMs**

### **Best Practices:**

**A. Schema Extraction Function:**
```python
def schema_to_llm_prompt(model: type[BaseModel]) -> str:
    schema = model.model_json_schema()
    # Build descriptive text with:
    # - Field names, types, descriptions
    # - Constraints (min, max, patterns, enums)
    # - Examples from Field(examples=[...])
    # - Business rules from docstrings or custom extraction
    return formatted_prompt
```

**B. Using Instructor Library:**
```python
import instructor
schema = instructor.generate_openai_schema(YourModel)
# Returns OpenAI-compatible function schema with descriptions
```

**C. Key Elements to Include:**
1. **Field descriptions** from `Field(description="...")`
2. **Constraints** (min/max values, patterns, required fields)
3. **Examples** from `Field(examples=[...])`
4. **Business rules** as plain text
5. **Complete examples** from `json_schema_extra`

**Example Output Format:**
```
# EvaluationCriteria
Create evaluation criteria for decision making.

## Fields:
- **criteria** (array): List of evaluation criteria [min items: 2]
- **context** (string): Context description [default: "General decision making"]
- **require_budget** (boolean): Whether budget criterion is required

## Business Rules:
1. Must have at least 2 criteria
2. Must include budget criterion if require_budget=True
3. Total weight must be between 5.0 and 50.0

## Examples:
{ "context": "Choosing a laptop", "criteria": [...] }
```

## 3. **Example Annotation in Pydantic**

### **Two Main Approaches:**

**A. Field-level Examples** (Pydantic 2.0+):
```python
name: str = Field(
    ...,
    examples=["Budget", "Quality", "Features"],  # Appears in JSON schema
    description="Criterion name"
)
```

**B. Model-level Examples:**
```python
model_config = ConfigDict(
    json_schema_extra={
        "examples": [
            {
                "context": "Example 1",
                "criteria": [...]
            },
            {
                "context": "Example 2", 
                "criteria": [...]
            }
        ]
    }
)
```

**C. Both appear in `model_json_schema()`:**
```python
schema = YourModel.model_json_schema()
# Field examples: schema["properties"]["field_name"]["examples"]
# Model examples: schema["examples"]
```

**Benefits:**
- Examples appear in OpenAPI/Swagger documentation
- Can be extracted for LLM prompts
- Provide guidance to both humans and LLMs

## 4. **FailureResponse Pattern for LLMs**

### **Recommended Pattern:**

```python
from typing import Literal, Union
from pydantic import BaseModel, Field

class FailureResponse(BaseModel):
    """Standardized failure response."""
    success: Literal[False] = Field(False)
    error_type: Literal["validation", "business_rule", "llm"] = Field(...)
    error_message: str = Field(...)
    suggestion: str = Field("Please check input and try again.")
    
    @classmethod
    def from_exception(cls, exc: Exception) -> 'FailureResponse':
        return cls(
            error_type="validation",
            error_message=str(exc),
            suggestion="Check input against requirements."
        )

class SuccessResponse(BaseModel):
    """Standardized success response."""
    success: Literal[True] = Field(True)
    data: dict = Field(...)
    message: Optional[str] = None

# Union type for polymorphic responses
LLMResponse = Union[SuccessResponse, FailureResponse]

# Usage pattern
def safe_llm_call(data: dict) -> LLMResponse:
    try:
        result = YourModel(**data)
        return SuccessResponse(data=result.model_dump())
    except Exception as e:
        return FailureResponse.from_exception(e)
```

### **Integration Strategies:**

1. **Wrapper Function:** `safe_create_model()` that returns `Union[Model, FailureResponse]`
2. **LLM Response Container:** Model that contains either success or failure
3. **Discriminated Union:** Using `success` field as discriminator
4. **Instructor Integration:** Wrap LLM calls with try/except returning FailureResponse

## **Practical Recommendations**

### **For Your Codebase:**

1. **Update `models.py`** to use `@model_validator` for business rules:
   - Add "must have >1 criteria" rule
   - Add "must include budget criterion" rule
   - Add total weight validation

2. **Create `schema_explanation.py`** with:
   - `schema_to_llm_prompt()` function
   - Business rule extraction from docstrings
   - Example inclusion from schema

3. **Add `responses.py`** with:
   - `FailureResponse` and `SuccessResponse` models
   - Helper methods for error conversion
   - Integration with existing LLM functions

4. **Enhance `EvaluationCriteria`** with:
   - `Field(examples=[...])` for common criterion names
   - `json_schema_extra` for complete examples
   - Better error messages in validators

### **Libraries to Consider:**

1. **`instructor`** (already in use): Excellent for structured LLM outputs
2. **`pydantic`** (already in use): Core validation and schema
3. **`pydantic-ai`**: Alternative to instructor with different API

### **Code Structure Suggestion:**

```
prompt-core/
├── prompt_core/
│   ├── models.py              # Enhanced with business rules
│   ├── responses.py           # FailureResponse, SuccessResponse
│   ├── schema_explanation.py  # LLM prompt generation
│   ├── llm_interaction.py     # Updated with error handling
│   └── cli.py
└── examples/
    └── enhanced_usage.py      # Showcase all features
```

## **Example Implementation for Your Project**

Based on your existing `EvaluationCriteria` model:

```python
# In models.py - Enhanced version
class EvaluationCriteria(BaseModel):
    criteria: List[Criterion] = Field(default_factory=list, min_length=2)
    context: str = Field(default="General decision making")
    require_budget: bool = Field(default=True)
    
    @model_validator(mode='after')
    def validate_business_rules(self):
        # Your specific rules
        if len(self.criteria) < 2:
            raise ValueError("Must have at least 2 criteria")
        
        if self.require_budget and not any(
            'budget' in c.name.lower() or 'cost' in c.name.lower() 
            for c in self.criteria
        ):
            raise ValueError("Must include a budget or cost criterion")
        
        return self
    
    # Add examples
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "context": "Birthday presents for 7-year-old",
                    "require_budget": True,
                    "criteria": [...]
                }
            ]
        }
    )
```

This approach gives you robust validation, clear LLM communication, and graceful error handling.
