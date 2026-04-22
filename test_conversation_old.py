#!/usr/bin/env python3
"""
Tests for conversation functionality.
"""
import sys
import unittest
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Optional

from prompt_core.models import EvaluationCriteria, Criterion
from prompt_core.conversation import ConversationAction, ConversationResult, ConversationOrchestrator


class TestConversationModels(unittest.TestCase):
    """Test the conversation models (no API calls)."""
    
    def test_conversation_action_validation(self):
        """Test ConversationAction validation based on action type."""
        # Valid continue action
        action = ConversationAction(action="continue", message="What's your budget?")
        self.assertEqual(action.action, "continue")
        self.assertEqual(action.message, "What's your budget?")
        self.assertIsNone(action.criteria)
        
        # Valid success action  
        criteria = EvaluationCriteria(
            context="test",
            criteria=[
                Criterion(name="budget", description="Budget constraint", weight=8.0),
                Criterion(name="quality", description="Quality level", weight=7.0),
            ]
        )
        action = ConversationAction(action="success", criteria=criteria)
        self.assertEqual(action.action, "success")
        self.assertIsNone(action.message)
        self.assertEqual(action.criteria, criteria)
        
        # Valid failure action
        action = ConversationAction(action="failure", message="Can't help with that")
        self.assertEqual(action.action, "failure")
        self.assertEqual(action.message, "Can't help with that")
        self.assertIsNone(action.criteria)
        
        # Invalid: continue without message
        with self.assertRaises(ValueError) as cm:
            ConversationAction(action="continue", message=None)
        self.assertIn("continue action requires message", str(cm.exception))
        
        # Invalid: success without criteria
        with self.assertRaises(ValueError) as cm:
            ConversationAction(action="success", criteria=None)
        self.assertIn("success action requires criteria", str(cm.exception))
        
        # Invalid: failure without message
        with self.assertRaises(ValueError) as cm:
            ConversationAction(action="failure", message=None)
        self.assertIn("failure action requires message", str(cm.exception))
    
    def test_conversation_result_factory_methods(self):
        """Test ConversationResult factory methods."""
        # Test continuing result
        result = ConversationResult.continuing("Please tell me more")
        self.assertEqual(result.message, "Please tell me more")
        self.assertIsNone(result.criteria)
        self.assertFalse(result.is_complete)
        
        # Test success result
        criteria = EvaluationCriteria(
            context="test",
            criteria=[
                Criterion(name="budget", description="Budget", weight=5.0),
                Criterion(name="features", description="Features", weight=6.0),
            ]
        )
        result = ConversationResult.success(criteria)
        self.assertEqual(result.message, "Criteria generated successfully!")
        self.assertEqual(result.criteria, criteria)
        self.assertTrue(result.is_complete)
        
        # Test failure result
        result = ConversationResult.failure("Maximum turns reached")
        self.assertEqual(result.message, "Failed: Maximum turns reached")
        self.assertIsNone(result.criteria)
        self.assertTrue(result.is_complete)


class TestConversationOrchestrator(unittest.TestCase):
    """Test ConversationOrchestrator with mocked LLM calls."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = Mock()
        self.mock_completion = Mock()
        self.mock_client.chat.completions.create.return_value = self.mock_completion
        
    def test_orchestrator_initialization(self):
        """Test ConversationOrchestrator initialization."""
        # Test with initial context
        orchestrator = ConversationOrchestrator(
            initial_context="birthday presents for child",
            max_turns=5,
            model="gpt-4o-mini"
        )
        
        self.assertEqual(orchestrator.turn_count, 0)
        self.assertEqual(orchestrator.max_turns, 5)
        self.assertEqual(orchestrator.model, "gpt-4o-mini")
        self.assertEqual(len(orchestrator.messages), 2)  # system + user message
        self.assertEqual(orchestrator.messages[0]["role"], "system")
        self.assertIn("birthday presents for child", orchestrator.messages[1]["content"])
        
        # Test without initial context
        orchestrator = ConversationOrchestrator()
        self.assertEqual(len(orchestrator.messages), 1)  # system only
        self.assertEqual(orchestrator.max_turns, 10)  # default
    
    @patch('prompt_core.llm_interaction.get_client')
    def test_process_turn_turn_limit(self, mock_get_client):
        """Test turn limit enforcement."""
        orchestrator = ConversationOrchestrator(max_turns=2)
        
        # First turn should work
        mock_get_client.return_value = self.mock_client
        self.mock_completion.action = "continue"
        self.mock_completion.message = "Question 1"
        
        result = orchestrator.process_turn("Hello")
        self.assertEqual(result.message, "Question 1")
        self.assertFalse(result.is_complete)
        self.assertEqual(orchestrator.turn_count, 1)
        
        # Second turn should work
        self.mock_completion.action = "continue"
        self.mock_completion.message = "Question 2"
        
        result = orchestrator.process_turn("Answer 1")
        self.assertEqual(result.message, "Question 2")
        self.assertFalse(result.is_complete)
        self.assertEqual(orchestrator.turn_count, 2)
        
        # Third turn should fail due to turn limit
        result = orchestrator.process_turn("Answer 2")
        self.assertEqual(result.message, "Failed: Maximum conversation turns (2) reached")
        self.assertTrue(result.is_complete)
        self.assertIsNone(result.criteria)
    
    @patch('prompt_core.llm_interaction.get_client')
    def test_process_turn_success(self, mock_get_client):
        """Test successful criteria generation in one turn."""
        orchestrator = ConversationOrchestrator(initial_context="test context")
        
        # Mock the LLM response
        mock_get_client.return_value = self.mock_client
        self.mock_completion.action = "success"
        
        # Create mock criteria
        mock_criteria = EvaluationCriteria(
            context="test context",
            criteria=[
                Criterion(name="budget", description="Budget constraint", weight=8.0),
                Criterion(name="quality", description="Quality level", weight=7.0),
            ]
        )
        self.mock_completion.criteria = mock_criteria
        self.mock_completion.message = None
        
        # Process turn
        result = orchestrator.process_turn("Let's create criteria")
        
        # Verify result
        self.assertTrue(result.is_complete)
        self.assertEqual(result.criteria, mock_criteria)
        self.assertEqual(result.message, "Criteria generated successfully!")
        
        # Verify messages were updated
        # system, initial user, user input (assistant message not added for success without message)
        self.assertEqual(len(orchestrator.messages), 3)
    
    @patch('prompt_core.llm_interaction.get_client')
    def test_process_turn_continue(self, mock_get_client):
        """Test continue action."""
        orchestrator = ConversationOrchestrator()
        
        mock_get_client.return_value = self.mock_client
        self.mock_completion.action = "continue"
        self.mock_completion.message = "What's your budget range?"
        self.mock_completion.criteria = None
        
        result = orchestrator.process_turn("Hello")
        
        self.assertFalse(result.is_complete)
        self.assertEqual(result.message, "What's your budget range?")
        self.assertIsNone(result.criteria)
        self.assertEqual(orchestrator.turn_count, 1)
    
    @patch('prompt_core.llm_interaction.get_client')
    def test_process_turn_failure(self, mock_get_client):
        """Test failure action from LLM."""
        orchestrator = ConversationOrchestrator()
        
        mock_get_client.return_value = self.mock_client
        self.mock_completion.action = "failure"
        self.mock_completion.message = "I don't have enough information to help"
        self.mock_completion.criteria = None
        
        result = orchestrator.process_turn("Something vague")
        
        self.assertTrue(result.is_complete)
        self.assertEqual(result.message, "Failed: I don't have enough information to help")
        self.assertIsNone(result.criteria)
    
    @patch('prompt_core.llm_interaction.get_client')
    def test_process_turn_empty_input(self, mock_get_client):
        """Test processing turn with empty input (starting conversation)."""
        orchestrator = ConversationOrchestrator(initial_context="test")
        
        mock_get_client.return_value = self.mock_client
        self.mock_completion.action = "continue"
        self.mock_completion.message = "First question"
        
        # Empty input should still add to messages if non-empty
        result = orchestrator.process_turn("")
        
        # Messages should have: system, initial user, assistant
        self.assertEqual(len(orchestrator.messages), 3)
    
    def test_extract_user_friendly_error(self):
        """Test error message extraction."""
        orchestrator = ConversationOrchestrator()
        
        # Test Pydantic validation error extraction
        error_msg = "ValueError: Must have at least 2 criteria"
        result = orchestrator._extract_user_friendly_error(error_msg)
        self.assertEqual(result, "Must have at least 2 criteria")
        
        # Test quoted error message extraction - the regex should extract content between quotes
        error_msg = 'Validation failed: "Must include budget criterion"'
        result = orchestrator._extract_user_friendly_error(error_msg)
        # Actually, this error doesn't contain "ValueError" or "validation error" 
        # so it falls through to generic error handler
        # Let's test with a proper validation error
        error_msg = 'Validation error: "Must include budget criterion"'
        result = orchestrator._extract_user_friendly_error(error_msg)
        self.assertEqual(result, "Must include budget criterion")
        
        # Test maximum retries error
        error_msg = "Maximum retries (3) exceeded"
        result = orchestrator._extract_user_friendly_error(error_msg)
        self.assertEqual(result, "Could not get valid response from AI after multiple attempts")
        
        # Test API key error
        error_msg = "Authentication error: Invalid API key"
        result = orchestrator._extract_user_friendly_error(error_msg)
        self.assertEqual(result, "API authentication error - check your API key")
        
        # Test connection error
        error_msg = "Connection timeout after 30 seconds"
        result = orchestrator._extract_user_friendly_error(error_msg)
        self.assertEqual(result, "Connection error - check your network")
        
        # Test generic error fallback
        error_msg = "Some random error " + "x" * 200
        result = orchestrator._extract_user_friendly_error(error_msg)
        self.assertTrue(result.startswith("Error: Some random error"))


class TestBusinessRules(unittest.TestCase):
    """Test business rule validation in models."""
    
    def test_evaluation_criteria_business_rules(self):
        """Test business rules for EvaluationCriteria."""
        # Valid criteria with "budget" (lowercase)
        criteria = EvaluationCriteria(
            context="test",
            criteria=[
                Criterion(name="budget", description="Budget constraint", weight=8.0),
                Criterion(name="quality", description="Quality level", weight=7.0),
            ]
        )
        # Should not raise error
        
        # Valid criteria with "Budget" (uppercase)
        criteria = EvaluationCriteria(
            context="test",
            criteria=[
                Criterion(name="Budget", description="Budget constraint", weight=8.0),
                Criterion(name="quality", description="Quality level", weight=7.0),
            ]
        )
        # Should not raise error
        
        # Invalid: only 1 criterion
        with self.assertRaises(ValueError) as cm:
            criteria = EvaluationCriteria(
                context="test",
                criteria=[
                    Criterion(name="budget", description="Budget constraint", weight=8.0),
                ]
            )
        self.assertIn("Must have at least 2 criteria", str(cm.exception))
        
        # Invalid: no "budget" criterion (case-insensitive)
        with self.assertRaises(ValueError) as cm:
            criteria = EvaluationCriteria(
                context="test",
                criteria=[
                    Criterion(name="cost", description="Cost constraint", weight=8.0),
                    Criterion(name="quality", description="Quality level", weight=7.0),
                ]
            )
        self.assertIn("Must include a criterion named 'budget'", str(cm.exception))
        
        # Invalid: has "budget" but misspelled
        with self.assertRaises(ValueError) as cm:
            criteria = EvaluationCriteria(
                context="test",
                criteria=[
                    Criterion(name="budgte", description="Misspelled budget", weight=8.0),
                    Criterion(name="quality", description="Quality level", weight=7.0),
                ]
            )
        self.assertIn("Must include a criterion named 'budget'", str(cm.exception))


if __name__ == '__main__':
    unittest.main(verbosity=2)