#!/usr/bin/env python3
"""
Check API key configuration and diagnose test failures.
Run this script when tests fail due to missing API keys.
"""
import os
import sys
from pathlib import Path

def check_api_keys():
    """Check which API keys are configured."""
    print("🔍 Checking API key configuration...")
    print("-" * 50)
    
    # Check .env file
    env_path = Path(".env")
    if env_path.exists():
        print(f"✓ Found .env file at: {env_path.absolute()}")
        
        # Read .env to see what's configured
        try:
            with open(env_path, 'r') as f:
                env_content = f.read()
                
            # Check for API keys in .env
            keys_to_check = [
                ("OPENAI_API_KEY", "OpenAI"),
                ("GOOGLE_API_KEY", "Google Gemini"),
                ("GROQ_API_KEY", "Groq"),
                ("ANTHROPIC_API_KEY", "Anthropic Claude"),
                ("TOGETHER_API_KEY", "Together AI"),
            ]
            
            for key_name, provider_name in keys_to_check:
                if key_name in env_content:
                    # Check if it's just the template value
                    lines = env_content.split('\n')
                    for line in lines:
                        if line.startswith(f"{key_name}="):
                            value = line.split('=', 1)[1].strip()
                            if value and "your-" not in value and len(value) > 10:
                                print(f"  ✓ {provider_name} key configured in .env")
                            else:
                                print(f"  ✗ {provider_name} key in .env but appears to be placeholder")
                            break
        except Exception as e:
            print(f"  ✗ Error reading .env: {e}")
    else:
        print(f"✗ No .env file found at: {env_path.absolute()}")
        print(f"  Create one with: cp .env.example .env")
    
    print()
    print("Checking environment variables...")
    print("-" * 50)
    
    # Check environment variables
    env_vars = [
        ("OPENAI_API_KEY", "OpenAI"),
        ("GOOGLE_API_KEY", "Google Gemini"),
        ("GROQ_API_KEY", "Groq"),
    ]
    
    any_set = False
    for var_name, provider_name in env_vars:
        value = os.getenv(var_name)
        if value and len(value) > 10:
            print(f"✓ {provider_name} API key set in environment")
            any_set = True
        elif value:
            print(f"✗ {provider_name} API key in environment but appears invalid (too short)")
        else:
            print(f"  {provider_name} API key not in environment")
    
    if not any_set:
        print("\n❌ No API keys found in environment!")
    
    print()
    print("Provider availability check...")
    print("-" * 50)
    
    # Check if provider packages are installed
    try:
        import openai
        print("✓ OpenAI Python SDK installed")
    except ImportError:
        print("✗ OpenAI Python SDK not installed (pip install openai)")
    
    try:
        import instructor
        print("✓ Instructor library installed")
    except ImportError:
        print("✗ Instructor library not installed (pip install instructor)")
    
    print()
    print("Recommended actions:")
    print("-" * 50)
    
    if not any_set:
        print("1. Get a FREE API key:")
        print("   - Google Gemini (recommended): https://makersuite.google.com/app/apikey")
        print("   - Groq: https://console.groq.com")
        print("   - OpenAI: https://platform.openai.com/api-keys")
        print()
        print("2. Configure your key:")
        print("   cp .env.example .env")
        print("   # Edit .env and add your API key")
        print()
        print("3. Test configuration:")
        print("   python -m unittest tests.unit.test_llm_interaction.TestLLMInteraction.test_call_llm_success")
    else:
        print("✓ API key(s) configured")
        print("Run a test:")
        print("  python -m unittest tests.unit.test_llm_interaction.TestLLMInteraction.test_call_llm_success")
    
    print()
    print("Note: Some tests REQUIRE API keys and will fail without them.")
    print("This is intentional - it verifies the entire system works end-to-end.")

if __name__ == "__main__":
    check_api_keys()