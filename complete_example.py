#!/usr/bin/env python3
"""
Complete example demonstrating all Pydantic best practices for LLM applications.
"""
import json
from typing import List, Optional, Union, Literal
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


# ============================================================================
# 1. MODELS WITH BUSINESS RULES
# ============================================================================

class Criterion(BaseModel):
    """A single evaluation criterion with validation."""
    
    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Name of the criterion",
        examples=["Budget", "Quality", "Features", "Safety", "Durability"]
    )
    
    description: str = Field(
        ...,
        min_length=10,
        description="Detailed description of what this criterion measures",
        examples=[
            "Total cost including purchase price and maintenance",
            "How well the product meets quality standards",
            "Number and usefulness of available features"
        ]
    )
    
    weight: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Importance weight from 0.0 (not important) to 10.0 (critical)",
        examples=[5.0, 7.5, 9.0, 2.5]
    )
    
    @field_validator('name')
    @classmethod
    def validate_name_not_empty(cls, v: str) -> str:
        """Ensure name is not just whitespace."""
        if not v.strip():
            raise ValueError('Name cannot be empty or whitespace only')
        return v.strip()


class EvaluationCriteria(BaseModel):
    """Evaluation criteria with business rules."""
    
    criteria: List[Criterion] = Field(
        default_factory=list,
        min_length=1,
        description="List of evaluation criteria"
    )
    
    context: str = Field(
        default="General decision making",
        description="Context for these evaluation criteria"
    )
    
    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "context": "Choosing a laptop for software development",
                    "criteria": [
                        {
                            "name": "Budget",
                            "description": "Total cost including laptop, accessories, and warranty",
                            "weight": 8.0,
                            "is_budget_related": True
                        },
                        {
                            "name": "Performance",
                            "description": "CPU, RAM, and GPU performance for development tasks",
                            "weight": 9.5,
                            "is_budget_related": False
                        }
                    ]
                }
            ]
        }
    )
    
    @model_validator(mode='after')
    def validate_min_criteria(self) -> 'EvaluationCriteria':
        """Business rule: Must have at least 2 criteria for meaningful evaluation."""
        if len(self.criteria) < 2:
            raise ValueError('Must have at least 2 criteria for meaningful evaluation')
        return self
    
    @model_validator(mode='after')
    def validate_budget_requirement(self) -> 'EvaluationCriteria':
        """Business rule: Must include budget criterion if required."""
        has_budget = any(criterion.is_budget_related for criterion in self.criteria)
        if not has_budget:
            raise ValueError('Must include at least one budget-related criterion')
        return self
    
    @model_validator(mode='after')
    def validate_unique_names(self) -> 'EvaluationCriteria':
        """Business rule: Criterion names must be unique."""
        names = [criterion.name for criterion in self.criteria]
        if len(names) != len(set(names)):
            raise ValueError('Criterion names must be unique')
        return self
    
    @model_validator(mode='after')
    def validate_total_weight(self) -> 'EvaluationCriteria':
        """Business rule: Total weight must be reasonable."""
        total_weight = sum(criterion.weight for criterion in self.criteria)
        if total_weight < 5.0:
            raise ValueError(f'Total weight ({total_weight}) is too low. Criteria should have more weight.')
        if total_weight > 50.0:
            raise ValueError(f'Total weight ({total_weight}) is too high. Consider reducing weights.')
        return self
    
    @property
    def total_weight(self) -> float:
        """Total weight of all criteria."""
        return sum(criterion.weight for criterion in self.criteria)
    
    @property
    def normalized_weights(self) -> List[float]:
        """Normalized weights (sum to 1.0)."""
        total = self.total_weight
        if total == 0:
            return [0.0] * len(self.criteria)
        return [criterion.weight / total for criterion in self.criteria]


# ============================================================================
# 2. SCHEMA EXPLANATION FOR LLMS
# ============================================================================

def schema_to_llm_prompt(model: type[BaseModel]) -> str:
    """Convert Pydantic model to descriptive text for LLM prompts."""
    schema = model.model_json_schema()
    
    prompt = []
    prompt.append(f"# {schema.get('title', model.__name__)}")
    
    if 'description' in schema:
        prompt.append(f"\n{schema['description']}")
    
    prompt.append("\n## Structure")
    
    for field_name, field_schema in schema.get('properties', {}).items():
        field_desc = [f"- **{field_name}**"]
        
        if 'description' in field_schema:
            field_desc.append(f": {field_schema['description']}")
        
        if 'type' in field_schema:
            field_desc.append(f" (type: {field_schema['type']})")
        
        # Add constraints
        constraints = []
        constraint_map = {
            'minimum': 'min', 'maximum': 'max',
            'minLength': 'min length', 'maxLength': 'max length',
            'pattern': 'pattern', 'default': 'default'
        }
        
        for key, desc in constraint_map.items():
            if key in field_schema:
                constraints.append(f"{desc}: {field_schema[key]}")
        
        if 'examples' in field_schema and field_schema['examples']:
            examples = field_schema['examples'][:3]
            constraints.append(f"examples: {examples}")
        
        if constraints:
            field_desc.append(f" [{', '.join(constraints)}]")
        
        prompt.append(''.join(field_desc))
    
    # Add example from schema
    if 'examples' in schema and schema['examples']:
        prompt.append("\n## Example")
        example = schema['examples'][0]
        prompt.append("```json")
        prompt.append(json.dumps(example, indent=2))
        prompt.append("```")
    
    return '\n'.join(prompt)


# ============================================================================
# 3. FAILURE RESPONSE PATTERN
# ============================================================================

class FailureResponse(BaseModel):
    """Standardized failure response for LLMs."""
    
    success: Literal[False] = Field(
        False,
        description="Always False for failure responses"
    )
    
    error_type: Literal["validation", "business_rule", "parsing", "llm"] = Field(
        ...,
        description="Type of error that occurred"
    )
    
    error_message: str = Field(
        ...,
        description="Human-readable error message"
    )
    
    @classmethod
    def from_exception(cls, exc: Exception, error_type: str = "validation") -> 'FailureResponse':
        """Create a FailureResponse from an exception."""
        return cls(
            error_type=error_type,
            error_message=str(exc),
            suggestion="Please check your input against the requirements and try again."
        )


class SuccessResponse(BaseModel):
    """Standardized success response."""
    
    success: Literal[True] = Field(
        True,
        description="Always True for success responses"
    )
    
    data: dict = Field(
        ...,
        description="The successful response data"
    )
    
# Union type for polymorphic responses
LLMResponse = Union[SuccessResponse, FailureResponse]


def safe_create_criteria(data: dict) -> LLMResponse:
    """
    Safely create EvaluationCriteria from data.
    Returns SuccessResponse with criteria or FailureResponse with error.
    """
    try:
        criteria = EvaluationCriteria(**data)
        return SuccessResponse(
            data=criteria.model_dump(),
            message="Successfully created evaluation criteria"
        )
    except Exception as e:
        return FailureResponse.from_exception(e)


# ============================================================================
# 4. EXAMPLE USAGE
# ============================================================================

def main():
    print("=" * 80)
    print("PYDANTIC BEST PRACTICES DEMONSTRATION")
    print("=" * 80)
    
    # 1. Show schema explanation for LLM
    print("\n1. SCHEMA EXPLANATION FOR LLM:")
    print("-" * 40)
    llm_prompt = schema_to_llm_prompt(EvaluationCriteria)
    print(llm_prompt[:800] + "..." if len(llm_prompt) > 800 else llm_prompt)
    
    # 2. Test valid data
    print("\n\n2. TESTING VALID DATA:")
    print("-" * 40)
    
    valid_data = {
        "context": "Choosing a car for a family",
        "require_budget_criterion": True,
        "criteria": [
            {
                "name": "Budget",
                "description": "Total cost including purchase, insurance, and maintenance",
                "weight": 8.5,
                "is_budget_related": True
            },
            {
                "name": "Safety",
                "description": "Crash test ratings and safety features",
                "weight": 9.0,
                "is_budget_related": False
            },
            {
                "name": "Fuel Efficiency",
                "description": "Miles per gallon and environmental impact",
                "weight": 7.0,
                "is_budget_related": False
            }
        ]
    }
    
    response = safe_create_criteria(valid_data)
    if response.success:
        print("✓ Successfully created criteria")
        criteria = EvaluationCriteria(**response.data)
        print(f"  Context: {criteria.context}")
        print(f"  Number of criteria: {len(criteria.criteria)}")
        print(f"  Total weight: {criteria.total_weight}")
        print(f"  Has budget criterion: {any(c.is_budget_related for c in criteria.criteria)}")
    else:
        print(f"✗ Failed: {response.error_message}")
    
    # 3. Test invalid data (business rule violation)
    print("\n\n3. TESTING INVALID DATA (BUSINESS RULE VIOLATION):")
    print("-" * 40)
    
    invalid_data = {
        "context": "Test",
        "require_budget_criterion": True,
        "criteria": [
            {
                "name": "Quality",
                "description": "Product quality assessment",
                "weight": 5.0,
                "is_budget_related": False  # Missing budget criterion!
            }
        ]
    }
    
    response = safe_create_criteria(invalid_data)
    if response.success:
        print("✗ Should have failed (missing budget criterion)")
    else:
        print(f"✓ Correctly failed with: {response.error_message}")
        print(f"  Error type: {response.error_type}")
        print(f"  Suggestion: {response.suggestion}")
    
    # 4. Test another invalid case (too few criteria)
    print("\n\n4. TESTING TOO FEW CRITERIA:")
    print("-" * 40)
    
    few_criteria_data = {
        "context": "Test",
        "require_budget_criterion": False,
        "criteria": [
            {
                "name": "Only One",
                "description": "Only one criterion",
                "weight": 5.0,
                "is_budget_related": False
            }
        ]
    }
    
    response = safe_create_criteria(few_criteria_data)
    if response.success:
        print("✗ Should have failed (need at least 2 criteria)")
    else:
        print(f"✓ Correctly failed with: {response.error_message}")
    
    # 5. Show JSON schema with examples
    print("\n\n5. JSON SCHEMA WITH EXAMPLES:")
    print("-" * 40)
    
    schema = EvaluationCriteria.model_json_schema()
    print("Schema keys:", list(schema.keys()))
    print("\nProperties:", list(schema.get('properties', {}).keys()))
    
    if 'examples' in schema:
        print(f"\nNumber of examples in schema: {len(schema['examples'])}")
    
    # 6. Field examples
    print("\n\n6. FIELD-LEVEL EXAMPLES:")
    print("-" * 40)
    
    criterion_schema = Criterion.model_json_schema()
    for field_name, field_schema in criterion_schema.get('properties', {}).items():
        if 'examples' in field_schema:
            examples = field_schema['examples'][:2]
            print(f"{field_name}: {examples}")
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
