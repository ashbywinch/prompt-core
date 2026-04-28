from pydantic import BaseModel, Field, model_validator
from typing import List, Literal, Optional


class ConversationAction(BaseModel):
    """LLM's decision about conversation flow with discriminator field."""

    action: Literal["continue", "success", "failure"]
    message: Optional[str] = None
    criteria: Optional["EvaluationCriteria"] = None

    @model_validator(mode="after")
    def validate_action_consistency(self):
        if self.action in ["continue", "failure"] and not self.message:
            raise ValueError(f"{self.action} action requires message")
        if self.action == "success" and not self.criteria:
            raise ValueError("success action requires criteria")
        return self


class ConversationResult(BaseModel):
    """Result of a conversation turn."""

    criteria: Optional["EvaluationCriteria"] = None
    message: str
    is_complete: bool

    @classmethod
    def continuing(cls, message: str) -> "ConversationResult":
        return cls(criteria=None, message=message, is_complete=False)

    @classmethod
    def success(cls, criteria: "EvaluationCriteria") -> "ConversationResult":
        return cls(
            criteria=criteria,
            message="Criteria generated successfully!",
            is_complete=True,
        )

    @classmethod
    def failure(cls, message: str) -> "ConversationResult":
        return cls(criteria=None, message=f"Failed: {message}", is_complete=True)


class Criterion(BaseModel):
    """A single criterion for evaluating options."""

    name: str = Field(
        ..., description="Name of the criterion (e.g., 'Educational Value', 'Safety')"
    )
    description: str = Field(
        ..., description="Detailed description of what this criterion measures"
    )
    weight: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Importance weight from 0.0 (not important) to 10.0 (critical)",
    )
    ideal_value: Optional[str] = Field(
        default=None,
        description="Ideal or target value for this criterion (e.g., 'High quality materials', 'Under $50')",
    )


class EvaluationCriteria(BaseModel):
    """A list of criteria for evaluating possible choices."""

    criteria: List[Criterion] = Field(
        default_factory=list, description="List of evaluation criteria"
    )
    context: str = Field(
        default="General decision making",
        description="Context for these evaluation criteria (e.g., 'Birthday presents for a 7-year-old child')",
    )

    @model_validator(mode="after")
    def validate_business_rules(self):
        """Validate business rules for EvaluationCriteria."""
        from .exceptions import CriteriaValidationError

        # Rule 1: Must have at least 2 criteria
        if len(self.criteria) < 2:
            raise CriteriaValidationError("Must have at least 2 criteria")

        # Rule 2: Must include "budget" criterion (case-insensitive)
        if not any(c.name.lower() == "budget" for c in self.criteria):
            raise CriteriaValidationError(
                "Must include a criterion named 'budget' (case-insensitive)"
            )
        return self

    def add_criterion(
        self,
        name: str,
        description: str,
        weight: float = 1.0,
        ideal_value: Optional[str] = None,
    ):
        """Helper method to add a criterion."""
        self.criteria.append(
            Criterion(
                name=name,
                description=description,
                weight=weight,
                ideal_value=ideal_value,
            )
        )

    def total_weight(self) -> float:
        """Calculate the total weight of all criteria."""
        return sum(criterion.weight for criterion in self.criteria)

    def normalized_weights(self) -> List[float]:
        """Get normalized weights (sum to 1.0)."""
        total = self.total_weight()
        if total == 0:
            return [0.0] * len(self.criteria)
        return [criterion.weight / total for criterion in self.criteria]


ConversationAction.model_rebuild()
ConversationResult.model_rebuild()
