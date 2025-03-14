import requests
import os
import json
import re  # Add this import for regex functionality
from typing import Dict, Any, Optional, Union, List
import streamlit as st

class LLMClient:
    def __init__(self, api_key: Optional[str] = None, provider: str = "openai"):
        """Initialize the LLM client with an API key.
        
        Args:
            api_key: The API key to use (defaults to environment variable)
            provider: The LLM provider to use ('openai' or 'google')
        """
        self.provider = provider
        
        # Set up API keys
        if provider == "openai":
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            self.api_url = os.environ.get("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
            self.model = os.environ.get("OPENAI_MODEL", "gpt-4-turbo")
        elif provider == "google":
            self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
            # For Google, we'll use the SDK instead of direct API calls
            self.model = os.environ.get("GOOGLE_MODEL", "gemini-2.0-flash-thinking-exp-01-21")
            try:
                import google.generativeai as genai
                self.genai = genai
                self.genai.configure(api_key=self.api_key)
            except ImportError:
                st.error("Google Generative AI SDK not found. Please install it with: pip install google-generativeai")
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def complete(self, prompt: str, max_tokens: int = 2000) -> str:
        """Send a completion request to the LLM API.
        
        Args:
            prompt: The prompt to send
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            The generated text
        """
        if self.provider == "openai":
            return self._complete_openai(prompt, max_tokens)
        elif self.provider == "google":
            return self._complete_google(prompt, max_tokens)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _complete_openai(self, prompt: str, max_tokens: int) -> str:
        """Handle OpenAI API completions."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            st.error(f"OpenAI API error: {e}")
            return f"Error: {e}"
    
    def _complete_google(self, prompt: str, max_tokens: int) -> str:
        """Handle Google Gemini API completions with improved JSON handling."""
        try:
            # Enhance prompt for better JSON responses
            if "JSON" in prompt or "json" in prompt:
                # Add instructions to ensure proper JSON formatting
                enhanced_prompt = (
                    f"{prompt}\n\n"
                    "CRITICAL INSTRUCTION: Your response MUST be valid JSON with EXACTLY ONE of these root keys:\n"
                    "1. 'result' - for direct solutions\n"
                    "2. 'subtasks' - for breaking down tasks\n\n"
                    "DO NOT return JSON with 'task_description' as a root key - it should only be inside subtask objects.\n"
                    "DO NOT include explanations outside the JSON object.\n"
                    "Ensure all JSON keys use double quotes."
                )
            else:
                enhanced_prompt = prompt
                
            generation_config = {
                "max_output_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40
            }
            
            model = self.genai.GenerativeModel(
                model_name=self.model,
                generation_config=generation_config
            )
            
            response = model.generate_content(enhanced_prompt)
            
            # Try to identify and clean up JSON in the response
            if hasattr(response, 'text'):
                result = response.text
                # Check if the response contains a code block with JSON
                json_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", result)
                if json_block_match:
                    try:
                        # If we can parse it as JSON, return just the JSON part
                        json_content = json_block_match.group(1).strip()
                        # Validate it's proper JSON
                        json.loads(json_content)
                        return json_content
                    except json.JSONDecodeError:
                        # If it's not valid JSON, return the original
                        pass
                        
            return response.text
        except Exception as e:
            st.error(f"Google API error: {e}")
            return f"Error: {e}"

    def set_api_key(self, api_key: str) -> None:
        """Set the API key."""
        self.api_key = api_key
        if self.provider == "google":
            try:
                self.genai.configure(api_key=self.api_key)
            except AttributeError:
                pass  # genai might not be imported
    
    def set_model(self, model: str) -> None:
        """Set the model to use."""
        self.model = model
    
    def set_provider(self, provider: str) -> None:
        """Set the provider to use ('openai' or 'google')."""
        if provider not in ["openai", "google"]:
            raise ValueError(f"Unsupported provider: {provider}")
        
        old_provider = self.provider
        self.provider = provider
        
        # Re-initialize if changing providers
        if old_provider != provider and provider == "google":
            try:
                import google.generativeai as genai
                self.genai = genai
                self.genai.configure(api_key=self.api_key)
            except ImportError:
                st.error("Google Generative AI SDK not found. Please install it with: pip install google-generativeai")

    def get_available_models(self) -> List[str]:
        """Get available models for the current provider."""
        if self.provider == "openai":
            return ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
        elif self.provider == "google":
            return [
                "gemini-2.0-flash-thinking-exp-01-21",
                "gemini-2.0-pro-exp-02-05",
                "gemini-2.0-flash-lite",
                "gemini-2.0-flash",
                "gemini-2.0-flash-exp"
            ]
        else:
            return []
