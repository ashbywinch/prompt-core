import re

with open('tests/unit/test_llm_interaction.py', 'r') as f:
    content = f.read()

# Update import paths
content = re.sub(r"@patch\('prompt_core\.llm_interaction\.get_client'\)", 
                 r"@patch('prompt_core.conversation.get_provider')", content)

# Update mock variable names
content = re.sub(r'mock_get_client', 'mock_get_provider', content)
content = re.sub(r'mock_client\.chat\.completions\.create', 'mock_provider.create_structured_response', content)

print("Updated test file")
