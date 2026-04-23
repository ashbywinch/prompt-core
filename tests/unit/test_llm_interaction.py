#!/usr/bin/env python3
"""
Unit tests for LLM interaction logic.

IMPORTANT TEST PHILOSOPHY:
=========================
1. REAL API TESTS: Some tests use REAL LLM API calls (not mocks)
2. INTENTIONAL FAILURES: These tests are DESIGNED TO FAIL without API keys
3. INFRASTRUCTURE VALIDATION: Failures expose missing infrastructure early
4. DO NOT MOCK: Do not convert real API tests to mocks - fix by configuring API keys

Purpose: Verify our code works with actual LLMs, not just mocked responses.
If tests fail with "API key not provided", the FIX is to CONFIGURE API KEYS,
not to modify tests to use mocks.

See TESTING.md and scripts/check_api_keys.py for help.
"""
import unittest
from unittest.mock import Mock, patch, MagicMock

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
            
        def create(self, model=None, messages=None, response_model=None, max_retries=None, **kwargs):
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
            return ConversationAction(
                action="continue",
                message="Test question"
            )


class TestLLMInteraction(unittest.TestCase):
    """Test LLM interaction logic with mocked API calls."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Don't specify model - use default from configuration
        self.orchestrator = ConversationOrchestrator()
    
    # =========================================================================
    # REAL API TESTS (require API keys - will FAIL without)
    # =========================================================================
    # These tests make ACTUAL API calls to verify our code works with real LLMs.
    # DO NOT convert these to mocked tests - fix by configuring API keys.
    # =========================================================================
    
    def test_call_llm_success(self):
        """
        Test successful LLM call returns ConversationAction.
        
        ⚠️  REAL API TEST - REQUIRES API KEY  ⚠️
        ========================================
        This test makes ACTUAL API calls to verify:
        1. Our prompts work with real LLMs
        2. LLMs generate valid ConversationAction responses
        3. The entire pipeline works end-to-end
        
        EXPECTED BEHAVIOR:
        - With API key: Test passes if prompts work correctly
        - Without API key: Test FAILS with "API key not provided"
        
        DO NOT MOCK THIS TEST!
        If it fails, configure API keys (see scripts/check_api_keys.py)
        """
        # Call _call_llm - will use real OpenAIProvider via get_provider()
        # This requires OPENAI_API_KEY to be set
        # INTENTIONAL: This will FAIL without API key to expose missing infrastructure
        action = self.orchestrator._call_llm()
        
        # If we get here, API key is set and call succeeded
        # Verify we got a valid ConversationAction
        self.assertIsInstance(action, ConversationAction)
        self.assertIn(action.action, ["continue", "success", "failure"])
        
        # If action is "continue" or "failure", it should have a message
        if action.action in ["continue", "failure"]:
            self.assertIsNotNone(action.message)
            self.assertTrue(len(action.message) > 0)
        
        # If action is "success", it should have criteria
        if action.action == "success":
            self.assertIsNotNone(action.criteria)
            # Criteria should meet business rules
            self.assertGreaterEqual(len(action.criteria.criteria), 2)
            has_budget = any(c.name.lower() == "budget" for c in action.criteria.criteria)
            self.assertTrue(has_budget, "Criteria should include 'budget'")
    
    # =========================================================================
    # MOCKED TESTS (do not require API keys)
    # =========================================================================
    # These tests use mocks to simulate error cases and test logic
    # that would be difficult to test with real API calls.
    # =========================================================================
    
    @patch('prompt_core.llm_interaction.get_client')
    def test_call_llm_with_validation_error(self, mock_get_client):
        """Test _call_llm when LLM raises validation error."""
        mock_client = MockInstructorClient()
        mock_client.responses = [ValueError("Validation failed after 3 retries")]
        
        mock_get_client.return_value = mock_client
        
        # _call_llm should propagate the exception
        with self.assertRaises(ValueError) as context:
            self.orchestrator._call_llm()
        
        self.assertIn("Validation failed", str(context.exception))
    
    @patch('prompt_core.llm_interaction.get_client')
    def test_call_llm_with_api_error(self, mock_get_client):
        """Test _call_llm when API call fails."""
        mock_client = MockInstructorClient()
        mock_client.responses = [Exception("API error: Invalid API key")]
        
        mock_get_client.return_value = mock_client
        
        # _call_llm should propagate the exception
        with self.assertRaises(Exception) as context:
            self.orchestrator._call_llm()
        
        self.assertIn("API error", str(context.exception))
    
    @patch('prompt_core.llm_interaction.get_client')
    def test_call_llm_passes_correct_parameters(self, mock_get_client):
        """Test that _call_llm passes correct parameters to LLM."""
        mock_client = MockInstructorClient()
        expected_action = ConversationAction(
            action="continue",
            message="Test"
        )
        mock_client.responses = [expected_action]
        
        mock_get_client.return_value = mock_client
        
        # Add some messages to test they're passed
        self.orchestrator.messages = [
            {"role": "system", "content": "Test system prompt"},
            {"role": "user", "content": "Test user message"}
        ]
        
        action = self.orchestrator._call_llm()
        
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
    
    
    
    @patch('prompt_core.llm_interaction.get_client')
    def test_call_llm_with_max_retries(self, mock_get_client):
        """Test that max_retries=3 is passed to instructor."""
        # Create valid criteria for success action
        from prompt_core.models import EvaluationCriteria, Criterion
        valid_criteria = EvaluationCriteria(
            context="test",
            criteria=[
                Criterion(name="budget", description="Budget", weight=8.0),
                Criterion(name="quality", description="Quality", weight=7.0),
            ]
        )
        expected_action = ConversationAction(
            action="success",
            criteria=valid_criteria
        )
        
        mock_client = MockInstructorClient()
        mock_client.responses = [expected_action]
        mock_get_client.return_value = mock_client
        
        self.orchestrator._call_llm()
        
        # Verify max_retries parameter  
        _, _, _, max_retries = mock_client.last_call_args
        self.assertEqual(max_retries, 3)
    
    @patch('prompt_core.llm_interaction.get_client')
    def test_call_llm_message_history(self, mock_get_client):
        """Test that current message history is passed to LLM."""
        expected_action = ConversationAction(
            action="continue",
            message="Response"
        )
        
        mock_client = MockInstructorClient()
        mock_client.responses = [expected_action]
        mock_get_client.return_value = mock_client
        
        # Simulate conversation history
        self.orchestrator.messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "Let's create criteria"}
        ]
        
        self.orchestrator._call_llm()
        
        # Verify messages parameter includes full history
        _, messages, _, _ = mock_client.last_call_args
        self.assertEqual(messages, self.orchestrator.messages)
    
    def test_multi_turn_conversation_with_real_llm(self):
        """
        Test multi-turn conversation with real LLM.
        
        ⚠️  REAL API TEST - REQUIRES API KEY  ⚠️
        ========================================
        This test makes ACTUAL API calls to verify multi-turn conversations.
        Tests conversation flow, turn counting, and response handling with real LLM.
        
        EXPECTED BEHAVIOR:
        - With API key: Test passes if multi-turn conversation works
        - Without API key: Test FAILS with "API key not provided"
        
        DO NOT MOCK THIS TEST!
        If it fails, configure API keys (see scripts/check_api_keys.py)
        """
        
        orchestrator = ConversationOrchestrator(
            initial_context="choosing a birthday gift",
            max_turns=3
        )
        
        # First turn
        result1 = orchestrator.process_turn("Hello, I need help choosing a gift")
        
        # Should get a response
        self.assertIsNotNone(result1.message)
        
        # If we get "continue", we can test another turn
        if not result1.is_complete and result1.message:
            print(f"Turn 1 response: {result1.message[:50]}...")
            
            # Second turn - respond to LLM's question
            # For testing, give a generic response
            result2 = orchestrator.process_turn("Around $50 budget")
            
            self.assertIsNotNone(result2.message)
            
            if not result2.is_complete:
                print(f"Turn 2 response: {result2.message[:50]}...")
                
                # Third turn
                result3 = orchestrator.process_turn("For a 7-year-old who likes science")
                
                self.assertIsNotNone(result3.message)
                print(f"Turn 3 response: {result3.message[:50]}...")
        
        # At this point, conversation should either:
        # 1. Be complete (success or failure)
        # 2. Still continuing (if max_turns not reached)
        
        # Verify conversation state is consistent
        self.assertEqual(orchestrator.turn_count, orchestrator.max_turns or 3)
        
        # If we got criteria, verify it meets business rules
        if result1.criteria or (hasattr(result2, 'criteria') and result2.criteria) or (hasattr(result3, 'criteria') and result3.criteria):
            criteria = result1.criteria or result2.criteria or result3.criteria
            self.assertGreaterEqual(len(criteria.criteria), 2)
            has_budget = any(c.name.lower() == "budget" for c in criteria.criteria)
            self.assertTrue(has_budget, "Criteria should include 'budget'")


if __name__ == '__main__':
    unittest.main(verbosity=2)