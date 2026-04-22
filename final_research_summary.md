# Pydantic Best Practices for LLM Applications - Research Summary

## 1. Encoding Business Rules in Pydantic Models

### Best Practices:

#### **A. Field-level Constraints (Simplest)**
```python
from pydantic import BaseModel, Field

class SimpleCriterion(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    weight: float = Field(default=1.0, ge=0.0, le=10.0)
    is_required: bool = Field(default=False)
```

**When to use:** Simple numeric/string bounds, patterns, required fields.

#### **B. Field Validators**
```python
from pydantic import field_validator

class BudgetCriterion(BaseModel):
    budget_type: Literal["fixed", "range", "maximum"]
    amount: float = Field(..., gt=0)
    
    @field_validator('amount')
    @classmethod
    def validate_amount(cls, v: float) -> float:
        if v <= 0:
            raise ValueError('Amount must be positive')
        return v
```

**When to use:** Single-field validation with custom logic.

#### **C. Model Validators (Recommended for Business Rules)**
```python
from pydantic import model_validator

class EvaluationCriteria(BaseModel):
    criteria: List[Criterion] = Field(default_factory=list, min_length=1)
    must_include_budget: bool = Field(default=True)
    
    @model_validator(mode='after')
    def validate_business_rules(self) -> 'EvaluationCriteria':
        # Rule 1: Must have at least 2 criteria
        if len(self.criteria) < 2:
            raise ValueError('Must have at least 2 criteria')
        
        # Rule 2: Must include budget if required
        if self.must_include_budget:
            has_budget = any('budget' in c.name.lower() for c in self.criteria)
            if not has_budget:
                raise ValueError('Must include a budget criterion')
        
        # Rule 3: Total weight constraints
        total_weight = sum(c.weight for c in self.criteria)
        if total_weight < 5.0 or total_weight > 50.0:
            raise ValueError(f'Total weight {total_weight} outside acceptable range 5-50')
        
        return self
```

**When to use:** Cross-field validation, complex business rules, dependencies between fields.

#### **D. Custom Types with Validation**
```python
from typing import Annotated
from pydantic import AfterValidator

def validate_percentage(v: float) -> float:
    if not 0 <= v <= 100:
        raise ValueError('Must be between 0 and 100')
    return v

Percentage = Annotated[float, AfterValidator(validate_percentage)]

class Criterion(BaseModel):
    importance: Percentage = Field(...)
```

**When to use:** Reusable validation logic across multiple models.

#### **E. Property-based Validation**
```python
class EvaluationCriteria(BaseModel):
    criteria: List[Criterion] = Field(default_factory=list)
    
    @property
    def is_valid(self) -> bool:
        return (
            len(self.criteria) >= 2 and
            any('budget' in c.name.lower() for c in self.criteria) and
            5.0 <= sum(c.weight for c in self.criteria) <= 50.0
        )
    
    def validation_errors(self) -> List[str]:
        errors = []
        if len(self.criteria) < 2:
            errors.append("Need at least 2 criteria")
        # ... more checks
        return errors
```

**When to use:** Non-blocking validation, computed properties, validation that doesn't prevent object creation.

### **Recommendation:**
- **Start with field-level constraints** for simple rules
- **Use `@model_validator` for business rules** involving multiple fields
- **Create custom types** for reusable validation patterns
- **Use properties** for optional/computed validation

## 2. Explaining Pydantic Schemas to LLMs

### Best Practices:

#### **A. Schema Extraction Function**
```python
def schema_to_llm_prompt(model: type[BaseModel]) -> str:
    """Convert Pydantic schema to LLM-friendly description."""
    schema = model.model_json_schema()
    
    prompt = [f"# {schema.get('title', model.__name__)}"]
    
    if 'description' in schema:
        prompt.append(f"\n{schema['description']}")
    
    prompt.append("\n## Fields:")
    for field_name, field_schema in schema.get('properties', {}).items():
        field_info = [f"- **{field_name}**"]
        
        if 'description' in field_schema:
            field_info.append(f": {field_schema['description']}")
        
        if 'type' in field_schema:
            field_info.append(f" (type: {field_schema['type']})")
        
        # Add constraints
        constraints = []
        for key, desc in [
            ('minimum', 'min'), ('maximum', 'max'),
            ('minLength', 'min length'), ('maxLength', 'max length'),
            ('pattern', 'pattern'), ('enum', 'allowed values')
        ]:
            if key in field_schema:
                constraints.append(f"{desc}: {field_schema[key]}")
        
        if constraints:
            field_info.append(f" [{', '.join(constraints)}]")
        
        prompt.append(''.join(field_info))
    
    return '\n'.join(prompt)
```

#### **B. Using Instructor Library**
```python
import instructor
from instructor import OpenAISchema

# Instructor provides built-in schema generation
schema = instructor.generate_openai_schema(YourModel)
# Returns OpenAI-compatible function schema
```

#### **C. Including Examples in Prompt**
```python
def enrich_schema_with_examples(schema: dict) -> str:
    """Add examples from schema to prompt."""
    prompt_parts = []
    
    for field_name, field_schema in schema.get('properties', {}).items():
        if 'examples' in field_schema:
            examples = field_schema['examples'][:3]  # First 3 examples
            prompt_parts.append(f"  Examples for {field_name}: {examples}")
    
    return '\n'.join(prompt_parts)
```

### **Recommendation:**
- **Extract full schema** using `model_json_schema()`
- **Format for readability** with clear sections and bullet points
- **Include constraints and examples** in the prompt
- **Consider using Instructor** for OpenAI-compatible schemas

## 3. Example Annotation in Pydantic

### Best Practices:

#### **A. Field-level Examples**
```python
from pydantic import BaseModel, Field

class Product(BaseModel):
    name: str = Field(
        ...,
        examples=["Laptop", "Smartphone", "Tablet"],
        description="Product name"
    )
    price: float = Field(
        ...,
        gt=0,
        examples=[999.99, 699.99, 299.99],
        description="Price in USD"
    )
```

#### **B. Model-level Examples**
```python
from pydantic import ConfigDict

class User(BaseModel):
    username: str = Field(..., min_length=3)
    email: str = Field(..., pattern=r".+@.+\..+")
    
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "username": "johndoe",
                    "email": "john@example.com"
                },
                {
                    "username": "janedoe",
                    "email": "jane@example.org"
                }
            ]
        }
    )
```

#### **C. Extracting Examples for LLMs**
```python
def get_examples_from_schema(model: type[BaseModel]) -> List[dict]:
    """Extract examples from Pydantic schema."""
    schema = model.model_json_schema()
    examples = []
    
    # Get model-level examples
    if 'examples' in schema:
        examples.extend(schema['examples'])
    
    # You can also generate examples from field-level examples
    for field_name, field_schema in schema.get('properties', {}).items():
        if 'examples' in field_schema:
            # Field has examples - could use them to generate sample instances
            pass
    
    return examples
```

### **Recommendation:**
- **Use `Field(examples=[...])`** for field-level examples
- **Use `json_schema_extra` in Config** for complete example instances
- **Examples appear in JSON schema** and can be extracted for LLM prompts

## 4. FailureResponse Pattern for LLMs

### Best Practices:

#### **A. Basic FailureResponse Model**
```python
from pydantic import BaseModel, Field
from typing import Optional, Literal

class FailureResponse(BaseModel):
    """Standardized failure response for LLMs."""
    
    success: Literal[False] = Field(
        False,
        description="Always False for failure responses"
    )
    
    error_type: Literal["validation", "business_rule", "parsing"] = Field(
        ...,
        description="Type of error"
    )
    
    error_message: str = Field(
        ...,
        description="Human-readable error message"
    )
    
    suggestion: Optional[str] = Field(
        None,
        description="How to fix the error"
    )
    
    @classmethod
    def from_exception(cls, exc: Exception) -> 'FailureResponse':
        return cls(
            error_type="validation",
            error_message=str(exc),
            suggestion="Please check your input and try again."
        )
```

#### **B. SuccessResponse Model**
```python
class SuccessResponse(BaseModel):
    """Standardized success response."""
    
    success: Literal[True] = Field(
        True,
        description="Always True for success responses"
    )
    
    data: dict = Field(
        ...,
        description="Response data"
    )
    
    message: Optional[str] = Field(
        None,
        description="Optional success message"
    )
```

#### **C. Union Type for Polymorphic Responses**
```python
from typing import Union

# Using Literal discriminators
LLMResponse = Union[SuccessResponse, FailureResponse]

# Or using a wrapper model
class LLMResponseWrapper(BaseModel):
    response: LLMResponse
    
    @classmethod
    def success(cls, data: dict, message: str = None) -> 'LLMResponseWrapper':
        return cls(response=SuccessResponse(data=data, message=message))
    
    @classmethod
    def failure(cls, error_type: str, error_message: str) -> 'LLMResponseWrapper':
        return cls(response=FailureResponse(
            error_type=error_type,
            error_message=error_message
        ))
```

#### **D. Integration with Existing Models**
```python
def safe_create_model(model_class: type[BaseModel], data: dict) -> Union[BaseModel, FailureResponse]:
    """Try to create model instance, return FailureResponse on error."""
    try:
        return model_class(**data)
    except Exception as e:
        return FailureResponse.from_exception(e)

# Usage
result = safe_create_model(EvaluationCriteria, user_input)
if isinstance(result, FailureResponse):
    # Handle failure
    print(f"Error: {result.error_message}")
else:
    # Use the successfully created model
    criteria = result
```

### **Recommendation:**
- **Create standardized `FailureResponse`** with error_type, message, and suggestion
- **Use `SuccessResponse`** for successful completions  
- **Use Union types** for polymorphic LLM responses
- **Provide helper methods** like `from_exception()` for easy error wrapping
- **Integrate with try/except patterns** in your LLM interaction code

## Recommended Libraries

1. **`instructor`** - Excellent for structured LLM outputs with Pydantic
   - Built-in schema generation for multiple LLM providers
   - `OpenAISchema` base class for OpenAI function calling
   - `generate_openai_schema()`, `generate_anthropic_schema()`, etc.

2. **`pydantic`** - Core validation and schema definition
   - `Field()` for constraints and examples
   - `@field_validator`, `@model_validator` for custom validation
   - `model_json_schema()` for schema extraction

3. **`pydantic-ai`** - Alternative to instructor for LLM integration
   - Similar functionality with different API design

## Practical Implementation Example

```python
import instructor
from pydantic import BaseModel, Field, model_validator
from typing import List, Union, Literal

# 1. Define models with business rules
class Criterion(BaseModel):
    name: str = Field(..., min_length=1, examples=["Budget", "Quality", "Features"])
    weight: float = Field(1.0, ge=0.0, le=10.0)
    description: str = Field(..., min_length=10)

class EvaluationCriteria(BaseModel):
    criteria: List[Criterion] = Field(..., min_length=2)
    context: str = Field("General decision making")
    
    @model_validator(mode='after')
    def validate_business_rules(self):
        if len(self.criteria) < 2:
            raise ValueError("Must have at least 2 criteria")
        
        total_weight = sum(c.weight for c in self.criteria)
        if total_weight == 0:
            raise ValueError("Total weight cannot be zero")
        
        return self

# 2. Failure response pattern
class FailureResponse(BaseModel):
    success: Literal[False] = False
    error_type: str
    error_message: str
    suggestion: str = "Please check your input and try again."

# 3. Schema explanation for LLM
def explain_to_llm(model: type[BaseModel]) -> str:
    schema = model.model_json_schema()
    # Build descriptive prompt from schema
    return f"Use this schema: {schema}"

# 4. LLM interaction with error handling
import openai

def get_llm_response(prompt: str, model_class: type[BaseModel]) -> Union[BaseModel, FailureResponse]:
    client = instructor.patch(openai.OpenAI())
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            response_model=model_class,
        )
        return response
    except Exception as e:
        return FailureResponse(
            error_type="llm_error",
            error_message=str(e),
            suggestion="The LLM couldn't generate valid output. Please rephrase your request."
        )
```

## Key Takeaways

1. **Business Rules**: Use `@model_validator` for complex rules, field constraints for simple ones
2. **LLM Schema Explanation**: Extract schema with `model_json_schema()` and format descriptively
3. **Examples**: Use `Field(examples=[...])` and `json_schema_extra` for comprehensive examples
4. **Failure Responses**: Standardize error responses for consistent LLM error handling
5. **Libraries**: Leverage `instructor` for LLM integration and `pydantic` for validation

This approach ensures robust validation, clear LLM communication, and graceful error handling in LLM applications.
