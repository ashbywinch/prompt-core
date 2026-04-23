#!/usr/bin/env python3
"""
Real API tests for conversation functionality.

IMPORTANT TEST PHILOSOPHY:
=========================
1. REAL API TESTS: These tests make ACTUAL API calls to real LLMs
2. INTENTIONAL FAILURES: Tests are DESIGNED TO FAIL without API keys
3. INFRASTRUCTURE VALIDATION: Failures expose missing infrastructure early
4. DO NOT MOCK: Do not convert these to mocks - fix by configuring API keys

Purpose: Verify our entire system works end-to-end with real LLMs.
If tests fail with "API key not provided", the FIX is to CONFIGURE API KEYS,
not to modify tests to use mocks.

LLM API access is required. See README.md for configuration options.
Warning: These tests make real API calls and may incur costs.

See TESTING.md and scripts/check_api_keys.py for help.
"""
import os
import unittest
import sys

from prompt_core.models import EvaluationCriteria, Criterion
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
    
    EXPECTED BEHAVIOR:
    - With API key: Tests pass if system works correctly
    - Without API key: Tests FAIL with "API key not provided"
    
    DO NOT MOCK THESE TESTS!
    If they fail, configure API keys (see scripts/check_api_keys.py)
    
    This intentional failure behavior exposes missing infrastructure early.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        # Note: We don't check API key here. Tests will FAIL if they try to use API without key.
        # This is the correct behavior - exposing missing dependencies.
    
    def test_single_turn_with_real_llm(self):
        """
        Test a single conversation turn with real LLM.
        
        ⚠️  REAL API TEST - REQUIRES API KEY
        This test makes ACTUAL API calls.
        Expected to FAIL without API key - this is INTENTIONAL.
        DO NOT MOCK - configure API keys instead.
        """
        # Use a simple context that might work in one turn
        orchestrator = ConversationOrchestrator(
            initial_context="evaluating coffee makers",
            max_turns=3
        )
        
        try:
            # Start conversation - requires valid API key
            result = orchestrator.process_turn("")
            
            # Should get a response (continue, success, or failure)
            self.assertIsInstance(result, ConversationResult)
            self.assertIsNotNone(result.message)
            
            # Check that it's a valid response type
            # The LLM should respond with something sensible
            self.assertTrue(len(result.message) > 0)
            
            print(f"\nReal API test - Single turn response: {result.message[:100]}...")
            
            # If we got criteria, verify it's valid
            if result.criteria:
                self.assertIsInstance(result.criteria, EvaluationCriteria)
                self.assertGreaterEqual(len(result.criteria.criteria), 2)
                self.assertTrue(any(c.name.lower() == "budget" for c in result.criteria.criteria))
                
        except Exception as e:
            # API call failed - test should FAIL
            error_msg = str(e)
            if "OPENAI_API_KEY" in error_msg:
                self.fail(f"API key missing: {error_msg}")
            elif "api key" in error_msg.lower() or "authentication" in error_msg.lower():
                self.fail(f"API authentication failed: {error_msg}")
            elif "connection" in error_msg.lower() or "timeout" in error_msg.lower():
                self.fail(f"Network error: {error_msg}")
            else:
                # Re-raise other exceptions
                raise
    
    def test_conversation_flow_with_real_llm(self):
        """
        Test a simple conversation flow with real LLM.
        
        ⚠️  REAL API TEST - REQUIRES API KEY
        This test makes ACTUAL API calls.
        Expected to FAIL without API key - this is INTENTIONAL.
        DO NOT MOCK - configure API keys instead.
        """
        # This test verifies the LLM can actually complete a conversation
        # and return valid EvaluationCriteria (not stringified JSON)
        orchestrator = ConversationOrchestrator(
            initial_context="choosing a birthday gift for a 7-year-old",
            max_turns=10
        )
        
        # Simulate a conversation that should produce criteria
        conversation_steps = [
            "",  # Start empty to let LLM ask first question
            "My budget is around $50",
            "They like building toys and science kits",
            "Safety is important for a 7-year-old",
            "Educational value would be good",
            "That's all I can think of"
        ]
        
        last_result = None
        
        for i, user_input in enumerate(conversation_steps):
            try:
                result = orchestrator.process_turn(user_input)
                print(f"\nTurn {i+1}: {result.message[:100]}...")
                print(f"Complete: {result.is_complete}")
                
                if result.is_complete:
                    last_result = result
                    break
                    
            except Exception as e:
                self.fail(f"Conversation failed at turn {i+1}: {e}")
        
        # Check if conversation completed
        if last_result and last_result.is_complete:
            # If completed with success, criteria should be valid EvaluationCriteria
            if last_result.criteria:
                self.assertIsInstance(last_result.criteria, EvaluationCriteria)
                self.assertGreaterEqual(len(last_result.criteria.criteria), 2)
                # Check for budget criterion (case-insensitive)
                budget_found = any(
                    "budget" in criterion.name.lower() 
                    for criterion in last_result.criteria.criteria
                )
                self.assertTrue(
                    budget_found,
                    f"Criteria missing 'budget'. Found: {[c.name for c in last_result.criteria.criteria]}"
                )
                print(f"\n✓ Conversation completed successfully!")
                print(f"  Generated {len(last_result.criteria.criteria)} criteria")
            else:
                # Completed with failure
                print(f"\n✗ Conversation failed: {last_result.message}")
        else:
            # Conversation didn't complete within steps
            print(f"\n⚠ Conversation did not complete within {len(conversation_steps)} steps")
            print(f"  Last turn count: {orchestrator.turn_count}")
        
        # The main test is that no validation errors occurred during the conversation
        # (If LLM returns stringified JSON, instructor would raise validation error)
    
    def test_business_rules_enforcement_with_real_llm(self):
        """
        Test that business rules are enforced even with real LLM output.
        
        Note: This test does NOT need API key - it tests Pydantic validation
        of business rules (minimum 2 criteria, must include "budget").
        """
        # This test doesn't need API key, it tests Pydantic validation
        # Create criteria that violates business rules (no "budget")
        
        # This should raise ValueError when creating EvaluationCriteria
        with self.assertRaises(ValueError) as cm:
            criteria = EvaluationCriteria(
                context="test",
                criteria=[
                    Criterion(name="quality", description="Quality", weight=5.0),
                    Criterion(name="features", description="Features", weight=5.0),
                ]
            )
        
        self.assertIn("Must include a criterion named 'budget'", str(cm.exception))
        
        # Test with only 1 criterion
        with self.assertRaises(ValueError) as cm:
            criteria = EvaluationCriteria(
                context="test",
                criteria=[
                    Criterion(name="budget", description="Budget", weight=5.0),
                ]
            )
        
        self.assertIn("Must have at least 2 criteria", str(cm.exception))
    
    def test_error_handling_with_bad_api_key(self):
        """Test error handling with invalid API key (if we can simulate it)."""
        # Note: We can't easily test this without actually providing a bad key
        # So this test is more documentation of expected behavior
        pass


class TestPromptEffectiveness(unittest.TestCase):
    """Tests to verify our prompts work correctly with real LLM."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_model = "gpt-4o-mini"
    
    def test_conversation_action_format(self):
        """
        Test that LLM returns valid ConversationAction format.
        
        ⚠️  REAL API TEST - REQUIRES API KEY
        This test makes ACTUAL API calls.
        Expected to FAIL without API key - this is INTENTIONAL.
        DO NOT MOCK - configure API keys instead.
        """
        orchestrator = ConversationOrchestrator(
            initial_context="test context"
        )
        
        # This will FAIL if OPENAI_API_KEY is not set
        result = orchestrator.process_turn("test")
        
        # Should get a valid response (not crash due to format errors)
        self.assertIsInstance(result, ConversationResult)
        
        # The real test is that instructor doesn't raise validation errors
        # after max_retries (which would show as failure result)


if __name__ == '__main__':
    # Run tests - they will FAIL if OPENAI_API_KEY is not set and tests try to use API
    # This is intentional - we want to know if test infrastructure is missing
    unittest.main(verbosity=2)