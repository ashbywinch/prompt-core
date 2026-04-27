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
        llm_response = orchestrator._call_llm()
        action = llm_response.content

        # Verify we got a valid ConversationAction
        self.assertIn(action.action, ["continue", "success", "failure"])
        # Verify usage metadata was captured
        self.assertIsNotNone(
            llm_response.usage, "LLMResponse should contain usage data"
        )

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

        result = orchestrator.run_conversation(
            [
                "Hello, I need help choosing a gift",
                "Around $50 budget",
                "For a 7-year-old who likes science",
            ]
        )

        self.assertIsNotNone(result.message)
        self.assertLessEqual(orchestrator.turn_count, orchestrator.max_turns)

        # If we got criteria, verify it meets business rules
        if result.criteria:
            self.assertGreaterEqual(len(result.criteria.criteria), 2)
            has_budget = any(
                c.name.lower() == "budget" for c in result.criteria.criteria
            )
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

        result = orchestrator.run_conversation(
            [
                "",
                "My budget is around $50",
                "They like building toys and science kits",
                "Safety is important for a 7-year-old",
                "Educational value would be good",
                "That's all I can think of",
            ]
        )

        # The conversation completed with criteria (not failure)
        self.assertIsNotNone(
            result.criteria,
            f"Conversation completed but with failure: {result.message}",
        )

        # Criteria should be valid EvaluationCriteria
        self.assertIsInstance(result.criteria, EvaluationCriteria)
        self.assertGreaterEqual(len(result.criteria.criteria), 2)

        # Check for budget criterion (case-insensitive)
        budget_found = any(
            "budget" in criterion.name.lower() for criterion in result.criteria.criteria
        )
        self.assertTrue(
            budget_found,
            f"Criteria missing 'budget'. Found: {[c.name for c in result.criteria.criteria]}",
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
