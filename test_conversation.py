#!/usr/bin/env python3
"""
Main test runner for conversation functionality.
Runs tests from the new test structure.
"""
import sys
import unittest

# Import all test modules from new structure
from tests.unit.test_models import (
    TestCriterionModel,
    TestEvaluationCriteriaModel,
    TestConversationActionModel,
    TestConversationResultModel
)
from tests.unit.test_orchestrator_logic import TestConversationOrchestratorLogic
from tests.unit.test_llm_interaction import TestLLMInteraction


def run_all_tests():
    """Run all unit tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestCriterionModel))
    suite.addTests(loader.loadTestsFromTestCase(TestEvaluationCriteriaModel))
    suite.addTests(loader.loadTestsFromTestCase(TestConversationActionModel))
    suite.addTests(loader.loadTestsFromTestCase(TestConversationResultModel))
    suite.addTests(loader.loadTestsFromTestCase(TestConversationOrchestratorLogic))
    suite.addTests(loader.loadTestsFromTestCase(TestLLMInteraction))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)