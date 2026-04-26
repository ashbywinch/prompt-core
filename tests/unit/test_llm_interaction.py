#!/usr/bin/env python3
"""
Unit tests for LLM interaction logic with mocked API calls.

These tests use mocks to simulate error cases and test logic
that would be difficult to test with real API calls.
"""

import unittest
from unittest.mock import patch

from prompt_core.conversation import ConversationOrchestrator, ConversationAction


class MockInstructorClient:
    """Mock instructor client for testing LLM interaction."""

    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_count = 0
        self.last_call_args = None
        self.last_call_kwargs = None
        self.chat = self.MockChatCompletions(self)

    class MockChatCompletions:
        def __init__(self, parent):
            self.parent = parent
            self.completions = self  # Allows chaining: client.chat.completions.create()

        def create(
            self,
            model=None,
            messages=None,
            response_model=None,
            max_retries=None,
            **kwargs,
        ):
            self.parent.call_count += 1
            self.parent.last_call_args = (model, messages, response_model, max_retries)
            self.parent.last_call_kwargs = kwargs

            if self.parent.responses:
                if len(self.parent.responses) > 0:
                    response = self.parent.responses.pop(0)
                    if isinstance(response, Exception):
                        raise response
                    return response
                else:
                    raise ValueError("No more responses in mock")

            # Default mock response
            return ConversationAction(action="continue", message="Test question")


class TestLLMInteraction(unittest.TestCase):
    """Test LLM interaction logic with mocked API calls."""

    def setUp(self):
        """Set up test fixtures."""
        # Don't specify model - use default from configuration
        self.orchestrator = ConversationOrchestrator()

    @patch("prompt_core.llm_interaction.get_client")
    def test_call_llm_with_validation_error(self, mock_get_client):
        """Test _call_llm when LLM raises validation error."""
        mock_client = MockInstructorClient()
        mock_client.responses = [ValueError("Validation failed after 3 retries")]

        mock_get_client.return_value = mock_client

        # _call_llm should propagate the exception
        with self.assertRaises(ValueError) as context:
            self.orchestrator._call_llm()

        self.assertIn("Validation failed", str(context.exception))

    @patch("prompt_core.llm_interaction.get_client")
    def test_call_llm_with_api_error(self, mock_get_client):
        """Test _call_llm when API call fails."""
        mock_client = MockInstructorClient()
        mock_client.responses = [Exception("API error: Invalid API key")]

        mock_get_client.return_value = mock_client

        # _call_llm should propagate the exception
        with self.assertRaises(Exception) as context:
            self.orchestrator._call_llm()

        self.assertIn("API error", str(context.exception))

    @patch("prompt_core.llm_interaction.get_client")
    def test_call_llm_passes_correct_parameters(self, mock_get_client):
        """Test that _call_llm passes correct parameters to LLM."""
        mock_client = MockInstructorClient()
        expected_action = ConversationAction(action="continue", message="Test")
        mock_client.responses = [expected_action]

        mock_get_client.return_value = mock_client

        # Add some messages to test they're passed
        self.orchestrator.messages = [
            {"role": "system", "content": "Test system prompt"},
            {"role": "user", "content": "Test user message"},
        ]

        self.orchestrator._call_llm()

        # Verify call parameters
        self.assertEqual(mock_client.call_count, 1)
        model, messages, response_model, max_retries = mock_client.last_call_args

        # Model should come from configuration
        from prompt_core.config import config

        expected_model = config.model
        self.assertEqual(model, expected_model)
        self.assertEqual(messages, self.orchestrator.messages)
        self.assertEqual(response_model, ConversationAction)
        self.assertEqual(max_retries, 3)
        self.assertEqual(
            mock_client.last_call_kwargs.get("timeout"),
            config.request_timeout_seconds,
        )

    @patch("prompt_core.llm_interaction.get_client")
    def test_call_llm_with_max_retries(self, mock_get_client):
        """Test that max_retries=3 is passed to instructor."""
        # Create valid criteria for success action
        from prompt_core.models import EvaluationCriteria, Criterion

        valid_criteria = EvaluationCriteria(
            context="test",
            criteria=[
                Criterion(name="budget", description="Budget", weight=8.0),
                Criterion(name="quality", description="Quality", weight=7.0),
            ],
        )
        expected_action = ConversationAction(action="success", criteria=valid_criteria)

        mock_client = MockInstructorClient()
        mock_client.responses = [expected_action]
        mock_get_client.return_value = mock_client

        self.orchestrator._call_llm()

        # Verify max_retries parameter
        _, _, _, max_retries = mock_client.last_call_args
        from prompt_core.config import config

        self.assertEqual(max_retries, 3)
        self.assertEqual(
            mock_client.last_call_kwargs.get("timeout"),
            config.request_timeout_seconds,
        )

    @patch("prompt_core.llm_interaction.get_client")
    def test_call_llm_message_history(self, mock_get_client):
        """Test that current message history is passed to LLM."""
        expected_action = ConversationAction(action="continue", message="Response")

        mock_client = MockInstructorClient()
        mock_client.responses = [expected_action]
        mock_get_client.return_value = mock_client

        # Simulate conversation history
        self.orchestrator.messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "Let's create criteria"},
        ]

        self.orchestrator._call_llm()

        # Verify messages parameter includes full history
        _, messages, _, _ = mock_client.last_call_args
        self.assertEqual(messages, self.orchestrator.messages)

    @patch("prompt_core.llm_interaction.get_client")
    def test_conversation_success_completion(self, mock_get_client):
        """Test that conversation can complete successfully with LLM returning success action."""
        # Create valid criteria for success action
        from prompt_core.models import EvaluationCriteria, Criterion

        valid_criteria = EvaluationCriteria(
            context="test conversation completion",
            criteria=[
                Criterion(name="budget", description="Budget constraint", weight=8.0),
                Criterion(
                    name="quality", description="Quality requirement", weight=7.0
                ),
            ],
        )
        success_action = ConversationAction(action="success", criteria=valid_criteria)

        mock_client = MockInstructorClient()
        mock_client.responses = [success_action]
        mock_get_client.return_value = mock_client

        # Process a turn - should return success result
        result = self.orchestrator.process_turn("Let's create criteria for test")

        # Verify success result
        self.assertTrue(result.is_complete)
        self.assertEqual(result.criteria, valid_criteria)
        self.assertIn("success", result.message.lower())

        # Verify LLM was called with correct parameters
        self.assertEqual(mock_client.call_count, 1)
        _, _, response_model, max_retries = mock_client.last_call_args
        self.assertEqual(response_model, ConversationAction)
        self.assertEqual(max_retries, 3)

    @patch("prompt_core.llm_interaction.get_client")
    def test_conversation_turn_limit_enforcement(self, mock_get_client):
        """Test that conversation stops after max_turns is reached."""
        # Mock continue actions to simulate extended conversation
        continue_action = ConversationAction(action="continue", message="Tell me more")
        mock_client = MockInstructorClient()
        # Create enough responses to exceed max_turns
        mock_client.responses = [
            continue_action
        ] * 20  # More than default max_turns (10)
        mock_get_client.return_value = mock_client

        # Process turns up to limit
        for i in range(10):  # Default max_turns is 10
            result = self.orchestrator.process_turn(f"User input {i}")
            self.assertFalse(result.is_complete)
            self.assertEqual(result.message, "Tell me more")

        # Next turn should raise TurnLimitExceededError
        from prompt_core.exceptions import TurnLimitExceededError

        with self.assertRaises(TurnLimitExceededError) as context:
            self.orchestrator.process_turn("One more turn")

        self.assertIn("10", str(context.exception))  # Error mentions max_turns

        # Verify exactly 10 calls were made (not 11)
        self.assertEqual(mock_client.call_count, 10)

    @patch("prompt_core.llm_interaction.get_client")
    def test_conversation_with_custom_max_turns(self, mock_get_client):
        """Test conversation with custom max_turns limit."""
        continue_action = ConversationAction(
            action="continue", message="Continue please"
        )
        mock_client = MockInstructorClient()
        mock_client.responses = [continue_action] * 5
        mock_get_client.return_value = mock_client

        # Create orchestrator with custom max_turns
        from prompt_core.conversation import ConversationOrchestrator

        orchestrator = ConversationOrchestrator(max_turns=3)

        # Process 3 turns should work
        for i in range(3):
            result = orchestrator.process_turn(f"Input {i}")
            self.assertFalse(result.is_complete)

        # 4th turn should fail
        from prompt_core.exceptions import TurnLimitExceededError

        with self.assertRaises(TurnLimitExceededError) as context:
            orchestrator.process_turn("Fourth turn")

        self.assertIn("3", str(context.exception))
        self.assertEqual(mock_client.call_count, 3)

    @patch("prompt_core.llm_interaction.get_client")
    def test_conversation_failure_action_termination(self, mock_get_client):
        """Test that LLM can terminate conversation with failure action."""
        # LLM explicitly says conversation failed
        failure_action = ConversationAction(
            action="failure", message="Cannot generate criteria with given information"
        )

        mock_client = MockInstructorClient()
        mock_client.responses = [failure_action]
        mock_get_client.return_value = mock_client

        # Process turn should raise ConversationFailedError
        from prompt_core.exceptions import ConversationFailedError

        with self.assertRaises(ConversationFailedError) as context:
            self.orchestrator.process_turn("Some user input")

        self.assertIn("Cannot generate criteria", str(context.exception))
        self.assertEqual(mock_client.call_count, 1)

    @patch("prompt_core.llm_interaction.get_client")
    def test_conversation_multiple_failure_attempts(self, mock_get_client):
        """Test conversation flow with multiple failure/continue attempts before success."""
        # Simulate: continue -> failure -> success
        continue_action = ConversationAction(
            action="continue", message="Need more info"
        )
        failure_action = ConversationAction(
            action="failure", message="Still not enough info"
        )

        from prompt_core.models import EvaluationCriteria, Criterion

        success_criteria = EvaluationCriteria(
            context="test",
            criteria=[
                Criterion(name="budget", description="Budget", weight=8.0),
                Criterion(name="quality", description="Quality", weight=7.0),
            ],
        )
        success_action = ConversationAction(action="success", criteria=success_criteria)

        mock_client = MockInstructorClient()
        mock_client.responses = [continue_action, failure_action, success_action]
        mock_get_client.return_value = mock_client

        # First turn: continue
        result1 = self.orchestrator.process_turn("First try")
        self.assertFalse(result1.is_complete)
        self.assertEqual(result1.message, "Need more info")

        # Second turn: failure - should raise
        from prompt_core.exceptions import ConversationFailedError

        with self.assertRaises(ConversationFailedError) as context:
            self.orchestrator.process_turn("Second try")

        self.assertIn("Still not enough info", str(context.exception))

        # Note: Conversation ends on failure, so success is never reached
        # This tests that failure action terminates conversation immediately

    @patch("prompt_core.llm_interaction.get_client")
    def test_validation_error_propagation(self, mock_get_client):
        """Test that validation errors from LLM propagate and terminate conversation."""
        # Simulate LLM returning invalid data that fails validation
        # Our mock client raises ValueError when validation fails
        mock_client = MockInstructorClient()
        mock_client.responses = [
            ValueError("Validation failed: Invalid JSON structure")
        ]
        mock_get_client.return_value = mock_client

        # Should propagate the validation error
        with self.assertRaises(ValueError) as context:
            self.orchestrator.process_turn("Test input")

        self.assertIn("Validation failed", str(context.exception))
        self.assertEqual(mock_client.call_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
