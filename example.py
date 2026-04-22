#!/usr/bin/env python3
"""
Example usage of prompt-core for generating evaluation criteria.
"""
import json
from prompt_core import generate_evaluation_criteria, chat_with_llm


def main():
    print("=== Prompt Core Example ===\n")
    
    # Example 1: Generate evaluation criteria for birthday presents
    print("1. Generating evaluation criteria for birthday presents...")
    context = "birthday presents for a 7-year-old child who loves science and art"
    
    criteria = generate_evaluation_criteria(context=context)
    
    print(f"\nContext: {criteria.context}")
    print(f"Number of criteria: {len(criteria.criteria)}")
    print(f"Total weight: {criteria.total_weight():.2f}")
    
    print("\nGenerated Criteria:")
    for i, criterion in enumerate(criteria.criteria, 1):
        print(f"\n{i}. {criterion.name} (weight: {criterion.weight})")
        print(f"   Description: {criterion.description}")
        if criterion.ideal_value:
            print(f"   Ideal: {criterion.ideal_value}")
    
    # Example 2: Chat with LLM about the criteria
    print("\n\n2. Chatting with LLM about the criteria...")
    
    criteria_summary = "\n".join([
        f"- {c.name}: {c.description}" for c in criteria.criteria
    ])
    
    chat_prompt = f"""
    Based on these evaluation criteria for {context}:
    {criteria_summary}
    
    Suggest 3 specific gift ideas that would score well across these criteria.
    For each gift, briefly explain why it would score well.
    """
    
    print("\nAsking LLM for gift suggestions...")
    response = chat_with_llm(chat_prompt)
    print(f"\nLLM Response:\n{response}")
    
    # Example 3: Save criteria to JSON
    print("\n\n3. Saving criteria to JSON file...")
    with open("evaluation_criteria.json", "w") as f:
        json.dump(criteria.model_dump(), f, indent=2)
    print("Saved to evaluation_criteria.json")
    
    # Example 4: Show normalized weights
    print("\n\n4. Normalized weights (sum to 1.0):")
    normalized = criteria.normalized_weights()
    for criterion, weight in zip(criteria.criteria, normalized):
        print(f"  {criterion.name}: {weight:.3f}")


if __name__ == "__main__":
    main()