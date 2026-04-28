#!/usr/bin/env python3
"""
Unit tests for Pydantic models.
No mocking needed - these test model validation and behavior.
"""

import unittest
from pydantic import ValidationError

from prompt_core.models import EvaluationCriteria, Criterion
from prompt_core.conversation import (
    ConversationAction,
    ConversationResult,
    CriteriaRefinementAction,
)
from prompt_core.exceptions import (
    CriteriaValidationError,
)


class TestCriterionModel(unittest.TestCase):
    """Test Criterion model."""

    def test_criterion_creation(self):
        """Test creating a Criterion."""
        criterion = Criterion(
            name="budget",
            description="Budget constraint",
            weight=8.0,
            ideal_value="Under $100",
        )

        self.assertEqual(criterion.name, "budget")
        self.assertEqual(criterion.description, "Budget constraint")
        self.assertEqual(criterion.weight, 8.0)
        self.assertEqual(criterion.ideal_value, "Under $100")

    def test_criterion_default_weight(self):
        """Test Criterion with default weight."""
        criterion = Criterion(name="quality", description="Quality level")

        self.assertEqual(criterion.name, "quality")
        self.assertEqual(criterion.weight, 1.0)  # default
        self.assertIsNone(criterion.ideal_value)

    def test_criterion_weight_constraints(self):
        """Test weight constraints (0.0 to 10.0)."""
        # Valid weights
        Criterion(name="test", description="test", weight=0.0)
        Criterion(name="test", description="test", weight=5.0)
        Criterion(name="test", description="test", weight=10.0)

        # Invalid weights should raise ValidationError (Pydantic's error)
        with self.assertRaises(ValidationError):
            Criterion(name="test", description="test", weight=-1.0)

        with self.assertRaises(ValidationError):
            Criterion(name="test", description="test", weight=11.0)


class TestEvaluationCriteriaModel(unittest.TestCase):
    """Test EvaluationCriteria model with business rules."""

    def setUp(self):
        """Set up test fixtures."""
        # Create valid criteria for tests
        self.valid_criteria = [
            Criterion(name="budget", description="Budget constraint", weight=8.0),
            Criterion(name="quality", description="Quality level", weight=7.0),
        ]

    def test_evaluation_criteria_creation(self):
        """Test creating valid EvaluationCriteria."""
        criteria = EvaluationCriteria(
            context="Choosing a laptop", criteria=self.valid_criteria
        )

        self.assertEqual(criteria.context, "Choosing a laptop")
        self.assertEqual(len(criteria.criteria), 2)
        self.assertEqual(criteria.criteria[0].name, "budget")
        self.assertEqual(criteria.criteria[1].name, "quality")

    def test_evaluation_criteria_default_context(self):
        """Test EvaluationCriteria with default context."""
        criteria = EvaluationCriteria(criteria=self.valid_criteria)

        self.assertEqual(criteria.context, "General decision making")
        self.assertEqual(len(criteria.criteria), 2)

    def test_business_rule_at_least_two_criteria(self):
        """Test business rule: must have at least 2 criteria."""
        # Valid: 2 criteria
        EvaluationCriteria(criteria=self.valid_criteria)

        # Valid: more than 2 criteria
        more_criteria = self.valid_criteria + [
            Criterion(name="features", description="Feature set", weight=6.0)
        ]
        EvaluationCriteria(criteria=more_criteria)

        # Invalid: only 1 criterion
        with self.assertRaises(CriteriaValidationError) as context:
            EvaluationCriteria(criteria=[self.valid_criteria[0]])

        self.assertIn("Must have at least 2 criteria", str(context.exception))

    def test_business_rule_must_include_budget(self):
        """Test business rule: must include 'budget' criterion (case-insensitive)."""
        # Valid: "budget" (lowercase)
        EvaluationCriteria(criteria=self.valid_criteria)

        # Valid: "Budget" (uppercase)
        budget_uppercase = [
            Criterion(name="Budget", description="Budget constraint", weight=8.0),
            Criterion(name="quality", description="Quality level", weight=7.0),
        ]
        EvaluationCriteria(criteria=budget_uppercase)

        # Valid: "BUDGET" (all caps)
        budget_allcaps = [
            Criterion(name="BUDGET", description="Budget constraint", weight=8.0),
            Criterion(name="quality", description="Quality level", weight=7.0),
        ]
        EvaluationCriteria(criteria=budget_allcaps)

        # Invalid: no budget criterion
        no_budget = [
            Criterion(name="cost", description="Cost constraint", weight=8.0),
            Criterion(name="quality", description="Quality level", weight=7.0),
        ]
        with self.assertRaises(CriteriaValidationError) as context:
            EvaluationCriteria(criteria=no_budget)

        self.assertIn("Must include a criterion named 'budget'", str(context.exception))

        # Invalid: misspelled budget
        misspelled = [
            Criterion(name="budgte", description="Misspelled budget", weight=8.0),
            Criterion(name="quality", description="Quality level", weight=7.0),
        ]
        with self.assertRaises(CriteriaValidationError) as context:
            EvaluationCriteria(criteria=misspelled)

        self.assertIn("Must include a criterion named 'budget'", str(context.exception))

    def test_add_criterion_method(self):
        """Test add_criterion helper method."""
        # Start with valid criteria
        criteria = EvaluationCriteria(criteria=self.valid_criteria)

        # Add another criterion
        criteria.add_criterion(
            name="features",
            description="Feature set",
            weight=6.0,
            ideal_value="Many useful features",
        )

        self.assertEqual(len(criteria.criteria), 3)
        self.assertEqual(criteria.criteria[2].name, "features")
        self.assertEqual(criteria.criteria[2].description, "Feature set")
        self.assertEqual(criteria.criteria[2].weight, 6.0)
        self.assertEqual(criteria.criteria[2].ideal_value, "Many useful features")

    def test_total_weight_method(self):
        """Test total_weight calculation."""
        criteria = EvaluationCriteria(
            criteria=[
                Criterion(name="budget", description="Budget", weight=8.0),
                Criterion(name="quality", description="Quality", weight=7.0),
                Criterion(name="features", description="Features", weight=6.0),
            ]
        )

        self.assertEqual(criteria.total_weight(), 21.0)

    def test_normalized_weights_method(self):
        """Test normalized_weights calculation."""
        criteria = EvaluationCriteria(
            criteria=[
                Criterion(name="budget", description="Budget", weight=8.0),
                Criterion(name="quality", description="Quality", weight=2.0),
            ]
        )

        normalized = criteria.normalized_weights()

        self.assertEqual(len(normalized), 2)
        self.assertAlmostEqual(normalized[0], 0.8)  # 8.0 / 10.0
        self.assertAlmostEqual(normalized[1], 0.2)  # 2.0 / 10.0
        self.assertAlmostEqual(sum(normalized), 1.0)

    def test_normalized_weights_zero_total(self):
        """Test normalized_weights with zero total weight."""
        criteria = EvaluationCriteria(
            criteria=[
                Criterion(name="budget", description="Budget", weight=0.0),
                Criterion(name="quality", description="Quality", weight=0.0),
            ]
        )

        normalized = criteria.normalized_weights()

        self.assertEqual(len(normalized), 2)
        self.assertEqual(normalized[0], 0.0)
        self.assertEqual(normalized[1], 0.0)


class TestConversationActionModel(unittest.TestCase):
    """Test ConversationAction model with discriminator field."""

    def test_conversation_action_continue(self):
        """Test continue action with message."""
        action = ConversationAction(action="continue", message="What's your budget?")

        self.assertEqual(action.action, "continue")
        self.assertEqual(action.message, "What's your budget?")
        self.assertIsNone(action.criteria)

    def test_conversation_action_success(self):
        """Test success action with criteria."""
        criteria = EvaluationCriteria(
            context="test",
            criteria=[
                Criterion(name="budget", description="Budget", weight=8.0),
                Criterion(name="quality", description="Quality", weight=7.0),
            ],
        )

        action = ConversationAction(action="success", criteria=criteria)

        self.assertEqual(action.action, "success")
        self.assertIsNone(action.message)
        self.assertEqual(action.criteria, criteria)

    def test_conversation_action_failure(self):
        """Test failure action with message."""
        action = ConversationAction(action="failure", message="Can't help with that")

        self.assertEqual(action.action, "failure")
        self.assertEqual(action.message, "Can't help with that")
        self.assertIsNone(action.criteria)

    def test_action_validation_continue_without_message(self):
        """Test that continue action requires message."""
        with self.assertRaises(ValueError) as context:
            ConversationAction(action="continue", message=None)

        self.assertIn("continue action requires message", str(context.exception))

    def test_action_validation_success_without_criteria(self):
        """Test that success action requires criteria."""
        with self.assertRaises(ValueError) as context:
            ConversationAction(action="success", criteria=None)

        self.assertIn("success action requires criteria", str(context.exception))

    def test_action_validation_failure_without_message(self):
        """Test that failure action requires message."""
        with self.assertRaises(ValueError) as context:
            ConversationAction(action="failure", message=None)

        self.assertIn("failure action requires message", str(context.exception))


class TestConversationResultModel(unittest.TestCase):
    """Test ConversationResult model and factory methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.valid_criteria = EvaluationCriteria(
            context="test",
            criteria=[
                Criterion(name="budget", description="Budget", weight=8.0),
                Criterion(name="quality", description="Quality", weight=7.0),
            ],
        )

    def test_continuing_factory_method(self):
        """Test ConversationResult.continuing() factory."""
        result = ConversationResult.continuing("Please tell me more")

        self.assertEqual(result.message, "Please tell me more")
        self.assertIsNone(result.criteria)
        self.assertFalse(result.is_complete)

    def test_success_factory_method(self):
        """Test ConversationResult.success() factory."""
        result = ConversationResult.success(self.valid_criteria)

        self.assertEqual(result.message, "Criteria generated successfully!")
        self.assertEqual(result.criteria, self.valid_criteria)
        self.assertTrue(result.is_complete)

    def test_failure_factory_method(self):
        """Test ConversationResult.failure() factory."""
        result = ConversationResult.failure("Maximum turns reached")

        self.assertEqual(result.message, "Failed: Maximum turns reached")
        self.assertIsNone(result.criteria)
        self.assertTrue(result.is_complete)

    def test_direct_creation(self):
        """Test direct ConversationResult creation."""
        result = ConversationResult(
            criteria=self.valid_criteria, message="Custom message", is_complete=True
        )

        self.assertEqual(result.criteria, self.valid_criteria)
        self.assertEqual(result.message, "Custom message")
        self.assertTrue(result.is_complete)


class TestCriteriaRefinementActionModel(unittest.TestCase):
    """Test refinement action model follows discriminator pattern."""

    def setUp(self):
        self.valid_criteria = EvaluationCriteria(
            context="test",
            criteria=[
                Criterion(name="budget", description="Budget", weight=8.0),
                Criterion(name="quality", description="Quality", weight=7.0),
            ],
        )

    def test_refinement_action_continue_requires_message(self):
        action = CriteriaRefinementAction(
            action="continue", message="What should change?"
        )
        self.assertEqual(action.action, "continue")
        self.assertEqual(action.message, "What should change?")
        self.assertIsNone(action.criteria)

    def test_refinement_action_success_requires_criteria(self):
        action = CriteriaRefinementAction(
            action="success", criteria=self.valid_criteria
        )
        self.assertEqual(action.action, "success")
        self.assertEqual(action.criteria, self.valid_criteria)

    def test_refinement_action_validation_failure_without_message(self):
        with self.assertRaises(ValueError) as context:
            CriteriaRefinementAction(action="failure", message=None)

        self.assertIn("failure action requires message", str(context.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
