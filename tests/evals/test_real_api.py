#!/usr/bin/env python3
"""
Real API tests (evals) for prompt-core.
These tests make ACTUAL API calls and require valid API keys.

Run with: make evals
"""

import unittest

from prompt_core.models import EvaluationCriteria
from prompt_core.conversation import ConversationOrchestrator, ConversationResult


class TestRealAPI(unittest.TestCase):
    """
    Integration tests that make real API calls.

    ⚠️  REAL API TESTS - ALL REQUIRE API KEYS  ⚠️
    ============================================
    These tests make ACTUAL API calls to verify:
    1. Our prompts work correctly with real LLMs
    2. The entire conversation flow works end-to-end
    3. Business rules are enforced with real LLM output
    """

    def test_call_llm_success(self):
        """
        Test successful LLM call returns ConversationAction.

        This test makes ACTUAL API calls to verify:
        1. Our prompts work with real LLMs
        2. LLMs generate valid ConversationAction responses
        """
        orchestrator = ConversationOrchestrator()

        # Call _call_llm - will use real provider
        action = orchestrator._call_llm()

        # Verify we got a valid ConversationAction
        self.assertIn(action.action, ["continue", "success", "failure"])

        # If action is "continue" or "failure", it should have a message
        if action.action in ["continue", "failure"]:
            self.assertIsNotNone(action.message)
            self.assertTrue(len(action.message) > 0)

        # If action is "success", it should have criteria
        if action.action == "success":
            self.assertIsNotNone(action.criteria)
            self.assertGreaterEqual(len(action.criteria.criteria), 2)
            has_budget = any(
                c.name.lower() == "budget" for c in action.criteria.criteria
            )
            self.assertTrue(has_budget, "Criteria should include 'budget'")

    def test_multi_turn_conversation_with_real_llm(self):
        """
        Test multi-turn conversation with real LLM.

        Tests conversation flow, turn counting, and response handling with real LLM.
        """
        orchestrator = ConversationOrchestrator(
            initial_context="choosing a birthday gift", max_turns=3
        )

        # First turn
        result1 = orchestrator.process_turn("Hello, I need help choosing a gift")

        # Should get a response
        self.assertIsNotNone(result1.message)

        # If we get "continue", we can test another turn
        if not result1.is_complete and result1.message:
            # Second turn
            result2 = orchestrator.process_turn("Around $50 budget")

            self.assertIsNotNone(result2.message)

            if not result2.is_complete:
                # Third turn
                result3 = orchestrator.process_turn(
                    "For a 7-year-old who likes science"
                )

                self.assertIsNotNone(result3.message)

        # Verify conversation state is consistent
        self.assertEqual(orchestrator.turn_count, orchestrator.max_turns or 3)

        # If we got criteria, verify it meets business rules
        if (
            result1.criteria
            or (hasattr(result2, "criteria") and result2.criteria)
            or (hasattr(result3, "criteria") and result3.criteria)
        ):
            criteria = result1.criteria or result2.criteria or result3.criteria
            self.assertGreaterEqual(len(criteria.criteria), 2)
            has_budget = any(c.name.lower() == "budget" for c in criteria.criteria)
            self.assertTrue(has_budget, "Criteria should include 'budget'")

    def test_single_turn_with_real_llm(self):
        """
        Test a single conversation turn with real LLM.
        """
        orchestrator = ConversationOrchestrator(
            initial_context="evaluating coffee makers", max_turns=3
        )

        result = orchestrator.process_turn("")

        # Should get a response (continue, success, or failure)
        self.assertIsInstance(result, ConversationResult)
        self.assertIsNotNone(result.message)

        # Check that it's a valid response type
        self.assertTrue(len(result.message) > 0)

        # If we got criteria, verify it's valid
        if result.criteria:
            self.assertIsInstance(result.criteria, EvaluationCriteria)
            self.assertGreaterEqual(len(result.criteria.criteria), 2)
            self.assertTrue(
                any(c.name.lower() == "budget" for c in result.criteria.criteria)
            )

    def test_conversation_flow_with_real_llm(self):
        """
        Test a simple conversation flow with real LLM.
        """
        orchestrator = ConversationOrchestrator(
            initial_context="choosing a birthday gift for a 7-year-old", max_turns=10
        )

        # Simulate a conversation that should produce criteria
        conversation_steps = [
            "",  # Start empty to let LLM ask first question
            "My budget is around $50",
            "They like building toys and science kits",
            "Safety is important for a 7-year-old",
            "Educational value would be good",
            "That's all I can think of",
        ]

        last_result = None

        for i, user_input in enumerate(conversation_steps):
            result = orchestrator.process_turn(user_input)

            if result.is_complete:
                last_result = result
                break

        # The conversation should complete within the steps
        self.assertTrue(
            last_result is not None and last_result.is_complete,
            f"Conversation did not complete within {len(conversation_steps)} steps.",
        )

        # If completed, it should be with success (not failure)
        self.assertIsNotNone(
            last_result.criteria,
            f"Conversation completed but with failure: {last_result.message}",
        )

        # Criteria should be valid EvaluationCriteria
        self.assertIsInstance(last_result.criteria, EvaluationCriteria)
        self.assertGreaterEqual(len(last_result.criteria.criteria), 2)

        # Check for budget criterion (case-insensitive)
        budget_found = any(
            "budget" in criterion.name.lower()
            for criterion in last_result.criteria.criteria
        )
        self.assertTrue(
            budget_found,
            f"Criteria missing 'budget'. Found: {[c.name for c in last_result.criteria.criteria]}",
        )

    def test_uncooperative_user_max_turns(self):
        """
        Test that conversation ends when user doesn't provide enough information.

        The conversation can end either by:
        1. Hitting max_turns limit (raises TurnLimitExceededError)
        2. LLM returning failure action (raises ConversationFailedError)

        Both are valid outcomes for uncooperative users.
        """
        orchestrator = ConversationOrchestrator(
            initial_context="choosing a laptop for programming", max_turns=3
        )

        # Uncooperative user responses
        uncooperative_responses = [
            "I'm not sure",
            "Maybe something good",
            "Whatever you think",
            "I don't know",
        ]

        last_exception = None
        completed = False

        for i, response in enumerate(uncooperative_responses):
            try:
                result = orchestrator.process_turn(response)

                if result.is_complete:
                    completed = True
                    break

            except Exception as e:
                last_exception = e
                break

        # Either the conversation completed with failure, or hit max turns
        self.assertTrue(
            completed or last_exception is not None,
            "Should have either completed or raised an exception",
        )

        # If it was a turn limit, verify the message
        if last_exception and "Maximum conversation turns" in str(last_exception):
            self.assertIn("3", str(last_exception))

    def test_conversation_action_format(self):
        """
        Test that LLM returns valid ConversationAction format.
        """
        orchestrator = ConversationOrchestrator(initial_context="test context")

        result = orchestrator.process_turn("test")

        # Should get a valid response (not crash due to format errors)
        self.assertIsInstance(result, ConversationResult)


if __name__ == "__main__":
    unittest.main(verbosity=2)
