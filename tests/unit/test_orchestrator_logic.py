#!/usr/bin/env python3
"""
Unit tests for ConversationOrchestrator logic.
These tests mock _call_llm() to test orchestrator behavior without real LLM calls.
"""
import unittest
from unittest.mock import Mock, patch

from prompt_core.models import EvaluationCriteria, Criterion
from prompt_core.conversation import ConversationOrchestrator, ConversationAction, ConversationResult


class TestConversationOrchestratorLogic(unittest.TestCase):
    """Test orchestrator logic with mocked LLM calls."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create valid criteria for tests
        self.valid_criteria = EvaluationCriteria(
            context="test context",
            criteria=[
                Criterion(name="budget", description="Budget constraint", weight=8.0),
                Criterion(name="quality", description="Quality level", weight=7.0),
            ]
        )
    
    def test_orchestrator_initialization(self):
        """Test ConversationOrchestrator initialization."""
        from prompt_core.config import config
        
        # Test with initial context
        orchestrator = ConversationOrchestrator(
            initial_context="birthday presents for child",
            max_turns=5
        )
        
        self.assertEqual(orchestrator.turn_count, 0)
        self.assertEqual(orchestrator.max_turns, 5)
        # Model should come from configuration
        expected_model = config.model
        self.assertEqual(orchestrator.model, expected_model)
        self.assertEqual(len(orchestrator.messages), 2)  # system + user message
        self.assertEqual(orchestrator.messages[0]["role"], "system")
        self.assertIn("birthday presents for child", orchestrator.messages[1]["content"])
        
        # Test without initial context
        orchestrator = ConversationOrchestrator()
        self.assertEqual(len(orchestrator.messages), 1)  # system only
        self.assertEqual(orchestrator.max_turns, 10)  # default
    
    @patch.object(ConversationOrchestrator, '_call_llm')
    def test_process_turn_success(self, mock_call_llm):
        """Test successful criteria generation in one turn."""
        orchestrator = ConversationOrchestrator(initial_context="test context")
        
        # Mock _call_llm to return success action
        action = ConversationAction(
            action="success",
            criteria=self.valid_criteria
        )
        mock_call_llm.return_value = action
        
        # Process turn
        result = orchestrator.process_turn("Let's create criteria")
        
        # Verify result
        self.assertTrue(result.is_complete)
        self.assertEqual(result.criteria, self.valid_criteria)
        self.assertEqual(result.message, "Criteria generated successfully!")
        self.assertEqual(orchestrator.turn_count, 1)
        
        # Verify messages: system, initial user, user input
        # (assistant message not added for success without message)
        self.assertEqual(len(orchestrator.messages), 3)
    
    @patch.object(ConversationOrchestrator, '_call_llm')
    def test_process_turn_continue(self, mock_call_llm):
        """Test continue action."""
        orchestrator = ConversationOrchestrator()
        
        # Mock _call_llm to return continue action
        action = ConversationAction(
            action="continue",
            message="What's your budget range?"
        )
        mock_call_llm.return_value = action
        
        result = orchestrator.process_turn("Hello")
        
        self.assertFalse(result.is_complete)
        self.assertEqual(result.message, "What's your budget range?")
        self.assertIsNone(result.criteria)
        self.assertEqual(orchestrator.turn_count, 1)
        
        # Verify messages: system, user input, assistant message
        self.assertEqual(len(orchestrator.messages), 3)
        self.assertEqual(orchestrator.messages[2]["role"], "assistant")
        self.assertEqual(orchestrator.messages[2]["content"], "What's your budget range?")
    
    @patch.object(ConversationOrchestrator, '_call_llm')
    def test_process_turn_failure_raises_exception(self, mock_call_llm):
        """Test that failure action from LLM raises exception."""
        orchestrator = ConversationOrchestrator()
        
        # Mock _call_llm to return failure action
        action = ConversationAction(
            action="failure",
            message="I don't have enough information to help"
        )
        mock_call_llm.return_value = action
        
        # process_turn() should raise ValueError for failure action
        with self.assertRaises(ValueError) as context:
            orchestrator.process_turn("Something vague")
        
        self.assertIn("LLM indicated failure: I don't have enough information to help", str(context.exception))
        self.assertEqual(orchestrator.turn_count, 1)  # Turn was counted
    
    @patch.object(ConversationOrchestrator, '_call_llm')
    def test_process_turn_empty_input(self, mock_call_llm):
        """Test processing turn with empty input (starting conversation)."""
        orchestrator = ConversationOrchestrator(initial_context="test")
        
        # Mock _call_llm to return continue action
        action = ConversationAction(
            action="continue",
            message="First question"
        )
        mock_call_llm.return_value = action
        
        # Empty input should still work
        result = orchestrator.process_turn("")
        
        self.assertFalse(result.is_complete)
        self.assertEqual(result.message, "First question")
        
        # Messages should have: system, initial user, assistant
        self.assertEqual(len(orchestrator.messages), 3)
    
    @patch.object(ConversationOrchestrator, '_call_llm')
    def test_multi_turn_conversation_sequence(self, mock_call_llm):
        """Test orchestrator handles sequence of turns correctly."""
        orchestrator = ConversationOrchestrator(max_turns=5)
        
        # Set up sequence of mock responses
        responses = [
            ConversationAction(action="continue", message="Question 1"),
            ConversationAction(action="continue", message="Question 2"),
            ConversationAction(action="success", criteria=self.valid_criteria)
        ]
        mock_call_llm.side_effect = responses
        
        # Turn 1: Continue
        result1 = orchestrator.process_turn("Hello")
        self.assertFalse(result1.is_complete)
        self.assertEqual(result1.message, "Question 1")
        self.assertEqual(orchestrator.turn_count, 1)
        
        # Turn 2: Continue
        result2 = orchestrator.process_turn("Answer 1")
        self.assertFalse(result2.is_complete)
        self.assertEqual(result2.message, "Question 2")
        self.assertEqual(orchestrator.turn_count, 2)
        
        # Turn 3: Success
        result3 = orchestrator.process_turn("Answer 2")
        self.assertTrue(result3.is_complete)
        self.assertEqual(result3.criteria, self.valid_criteria)
        self.assertEqual(orchestrator.turn_count, 3)
        
        # Verify total messages
        # system + user1 + assistant1 + user2 + assistant2 + user3
        # (no assistant message for success without message)
        self.assertEqual(len(orchestrator.messages), 6)
    
    @patch.object(ConversationOrchestrator, '_call_llm')
    def test_orchestrator_raises_on_max_turns(self, mock_call_llm):
        """Test orchestrator raises exception after max turns reached."""
        orchestrator = ConversationOrchestrator(max_turns=2)
        
        # Mock always returns continue
        mock_call_llm.return_value = ConversationAction(
            action="continue",
            message="Another question"
        )
        
        # First turn - OK
        result1 = orchestrator.process_turn("A1")
        self.assertFalse(result1.is_complete)
        self.assertEqual(orchestrator.turn_count, 1)
        
        # Second turn - OK
        result2 = orchestrator.process_turn("A2")
        self.assertFalse(result2.is_complete)
        self.assertEqual(orchestrator.turn_count, 2)
        
        # Third turn - should raise ValueError (turn limit)
        with self.assertRaises(ValueError) as context:
            orchestrator.process_turn("A3")
        
        self.assertIn("Maximum conversation turns (2) reached", str(context.exception))
    
    @patch.object(ConversationOrchestrator, '_call_llm')
    def test_process_turn_propagates_llm_exceptions(self, mock_call_llm):
        """Test that exceptions from _call_llm() propagate through process_turn()."""
        orchestrator = ConversationOrchestrator()
        
        # Mock _call_llm to raise an exception
        mock_call_llm.side_effect = ValueError("Validation failed after retries")
        
        # process_turn() should propagate the exception
        with self.assertRaises(ValueError) as context:
            orchestrator.process_turn("test")
        
        self.assertIn("Validation failed", str(context.exception))
        self.assertEqual(orchestrator.turn_count, 1)  # Turn was counted
    
    @patch.object(ConversationOrchestrator, '_call_llm')
    def test_invalid_action_raises_exception(self, mock_call_llm):
        """Test that invalid action from LLM raises exception."""
        orchestrator = ConversationOrchestrator()
        
        # Create a mock that looks like ConversationAction but has invalid action
        # This simulates what might happen if Pydantic validation somehow fails
        mock_action = Mock()
        mock_action.action = "invalid_action"  # Not in Literal["continue", "success", "failure"]
        mock_action.message = "test"
        mock_action.criteria = None
        mock_call_llm.return_value = mock_action
        
        # process_turn() should raise ValueError for invalid action
        with self.assertRaises(ValueError) as context:
            orchestrator.process_turn("test")
        
        self.assertIn("Invalid action received: invalid_action", str(context.exception))
    
    def test_turn_counting(self):
        """Test that turn counting works correctly."""
        orchestrator = ConversationOrchestrator(max_turns=3)
        
        self.assertEqual(orchestrator.turn_count, 0)
        
        # Simulate turns
        with patch.object(orchestrator, '_call_llm') as mock_call_llm:
            mock_call_llm.return_value = ConversationAction(
                action="continue",
                message="Test"
            )
            
            orchestrator.process_turn("Turn 1")
            self.assertEqual(orchestrator.turn_count, 1)
            
            orchestrator.process_turn("Turn 2")
            self.assertEqual(orchestrator.turn_count, 2)
            
            orchestrator.process_turn("Turn 3")
            self.assertEqual(orchestrator.turn_count, 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)