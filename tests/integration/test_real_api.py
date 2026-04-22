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

Required: OPENAI_API_KEY or other LLM provider API key
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
        # Use a cheaper/faster model for tests
        self.test_model = "gpt-4o-mini"
        
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
            model=self.test_model,
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
        # This is a simple test to see if the LLM understands the conversation format
        orchestrator = ConversationOrchestrator(
            initial_context="choosing a birthday gift for a 7-year-old",
            model=self.test_model,
            max_turns=5
        )
        
        # Start with empty input - this will FAIL if OPENAI_API_KEY is not set
        result1 = orchestrator.process_turn("")
        print(f"\nTurn 1: {result1.message[:100]}...")
        
        # The LLM should respond with either continue, success, or failure
        # If continue, we could simulate a simple response
        if not result1.is_complete and "budget" in result1.message.lower():
            # LLM is asking about budget - give a simple response
            result2 = orchestrator.process_turn("Around $50")
            print(f"Turn 2: {result2.message[:100]}...")
            
            if not result2.is_complete:
                # See what happens with another response
                result3 = orchestrator.process_turn("They like building toys")
                print(f"Turn 3: {result3.message[:100]}...")
        
        # At this point, we've tested that the LLM can engage in conversation
        # We don't assert specific outcomes because LLM responses are non-deterministic
    
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
            initial_context="test context",
            model=self.test_model
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