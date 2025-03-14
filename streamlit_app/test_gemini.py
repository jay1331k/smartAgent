#!/usr/bin/env python
"""
Test script for Google Gemini API
This helps users check if their API key is working correctly
"""
import os
import sys
import argparse

def test_gemini_api(api_key):
    """Test the Google Gemini API with the provided key."""
    try:
        import google.generativeai as genai
        print("✓ Successfully imported Google Generative AI module")
        
        # Configure the API
        print("Configuring Gemini with provided API key...")
        genai.configure(api_key=api_key)
        
        # List available models to verify connection
        print("Fetching available models...")
        models = genai.list_models()
        print(f"✓ Successfully connected to Gemini API. Available models:")
        
        for model in models:
            print(f"  - {model.name}")
        
        # Try a simple generation
        print("\nTesting a simple generation...")
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content("Hello, can you help me with coding?")
        print("\nGemini response:")
        print("-" * 50)
        print(response.text)
        print("-" * 50)
        
        print("\n✓ API key is working correctly!")
        return True
    except Exception as e:
        print(f"\n❌ Error while testing Gemini API: {str(e)}")
        print("\nPossible issues:")
        print("1. The API key might be incorrect or invalid")
        print("2. Your internet connection might have issues")
        print("3. The Gemini API might be experiencing downtime")
        print("4. Your account might have reached its quota limit")
        print("\nPlease check your API key and try again.")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test Google Gemini API")
    parser.add_argument("--key", type=str, help="Google Gemini API Key")
    
    args = parser.parse_args()
    
    # Get API key from command line args or prompt user
    api_key = args.key
    if not api_key:
        api_key = input("Please enter your Google Gemini API Key: ")
    
    test_gemini_api(api_key)

if __name__ == "__main__":
    main()
