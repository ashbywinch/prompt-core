#!/usr/bin/env python3
"""
Basic tests for core models.
These tests don't require any mocking or API access.
"""
import sys
import unittest

from tests.unit.test_models import (
    TestCriterionModel,
    TestEvaluationCriteriaModel,
    TestConversationActionModel,
    TestConversationResultModel
)


def run_basic_tests():
    """Run basic model tests (no mocking needed)."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add basic test classes (no mocking)
    suite.addTests(loader.loadTestsFromTestCase(TestCriterionModel))
    suite.addTests(loader.loadTestsFromTestCase(TestEvaluationCriteriaModel))
    suite.addTests(loader.loadTestsFromTestCase(TestConversationActionModel))
    suite.addTests(loader.loadTestsFromTestCase(TestConversationResultModel))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_basic_tests()
    sys.exit(0 if success else 1)