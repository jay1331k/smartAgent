#!/usr/bin/env python
"""
Helper script to guide users in getting a Google API key for Gemini
"""
import webbrowser
import os

def main():
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                Get Google API Key for Gemini                      ║
╚═══════════════════════════════════════════════════════════════════╝

Follow these steps to obtain a Google API key:

1. Go to Google AI Studio website (opening browser)
2. Sign in with your Google account
3. Click on "Get API key" in the top menu
4. Create a new API key or use an existing one
5. Copy the API key
6. Paste it into the Cline Web IDE sidebar

Press Enter to open the Google AI Studio website...
""")
    input()
    
    # Open Google AI Studio website
    webbrowser.open("https://makersuite.google.com/app/apikey")
    
    print("""
Browser opened. Once you have your API key:

1. Start the Cline Web IDE with:
   python run.py --no-auth

2. Paste your API key in the sidebar
3. Start using the AI assistant

Enjoy using Cline Web IDE with Google Gemini!
""")

if __name__ == "__main__":
    main()
