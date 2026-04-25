#!/usr/bin/env python3
"""
Unit tests for LLM interaction logic with mocked API calls.

These tests use mocks to simulate error cases and test logic
that would be difficult to test with real API calls.
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


if __name__ == '__main__':
    unittest.main(verbosity=2)