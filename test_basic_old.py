#!/usr/bin/env python3
"""
Basic test without API calls to verify the structure works.
"""
import json
from prompt_core.models import EvaluationCriteria, Criterion


def test_manual_creation():
    """Test creating criteria manually."""
    print("Testing manual creation of EvaluationCriteria...")
    
    # Create criteria with initial values (must include "budget" and at least 2 criteria)
    criteria = EvaluationCriteria(
        context="birthday presents for a 8-year-old",
        criteria=[
            Criterion(
                name="budget",
                description="Cost of the gift",
                weight=8.0,
                ideal_value="Under $50"
            ),
            Criterion(
                name="Educational Value",
                description="How much the gift teaches or develops skills",
                weight=8.0,
                ideal_value="High educational content"
            )
        ]
    )
    
    # Add more criteria
    criteria.add_criterion(
        name="Safety",
        description="Appropriateness for child's age and safety features",
        weight=9.5,
        ideal_value="Age-appropriate, non-toxic materials"
    )
    
    criteria.add_criterion(
        name="Entertainment Value",
        description="How fun and engaging the gift is",
        weight=7.5,
        ideal_value="Highly engaging and replayable"
    )
    
    print(f"Context: {criteria.context}")
    print(f"Number of criteria: {len(criteria.criteria)}")
    print(f"Total weight: {criteria.total_weight()}")
    
    print("\nCriteria:")
    for i, c in enumerate(criteria.criteria, 1):
        print(f"{i}. {c.name}: {c.description}")
        print(f"   Weight: {c.weight}, Ideal: {c.ideal_value}")
    
    # Test normalized weights
    print("\nNormalized weights:")
    normalized = criteria.normalized_weights()
    for c, w in zip(criteria.criteria, normalized):
        print(f"  {c.name}: {w:.3f} ({w*100:.1f}%)")
    
    # Test JSON serialization
    print("\nJSON serialization test...")
    json_data = criteria.model_dump()
    print(f"JSON keys: {list(json_data.keys())}")
    
    # Test deserialization
    criteria2 = EvaluationCriteria(**json_data)
    print(f"Deserialized successfully: {len(criteria2.criteria)} criteria")
    
    return criteria


if __name__ == "__main__":
    test_manual_creation()
    print("\n✓ All basic tests passed (without API calls)")