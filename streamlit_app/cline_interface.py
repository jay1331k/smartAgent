import streamlit as st
import os
import json
import time
import google.generativeai as genai
import sys
import re
from pathlib import Path

class ClineInterface:
    def __init__(self):
        if 'client' not in st.session_state:
            st.session_state.client = None
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
    
    def set_api_key(self, api_key):
        """Set up the client with the provided API key."""
        if api_key:
            try:
                # Configure the Gemini API
                genai.configure(api_key=api_key)
                st.session_state.client = genai
                # Add a test message to verify configuration
                print("Google Gemini API configured successfully.")
                return True
            except Exception as e:
                st.error(f"Error initializing client: {str(e)}")
                return False
        return False
    
    def generate_system_prompt(self, project_path):
        """Generate system prompt with project context."""
        system_prompt = """You are Cline, an AI programming assistant.
You help users build software projects by understanding their requirements and providing practical code solutions.
You can assist with code writing, debugging, and explaining concepts.

For file operations, tell the user to use the File Explorer panel.
For running commands, tell the user to use the Terminal panel.

Follow these guidelines:
1. Keep your answers focused on the user's programming needs
2. Provide complete, working code solutions
3. Explain your reasoning clearly
4. When suggesting file changes, specify the exact file path and code to be modified
5. Help the user understand errors and debugging strategies

The user is working in a web-based IDE with:
- A file explorer panel for browsing and editing files
- A code editor with syntax highlighting
- A terminal for running commands
- This chat interface to communicate with you

The user can download their complete project when finished.
"""
        
        # Add project files information
        files_info = "\n\nCurrent Project Structure:\n"
        for root, dirs, files in os.walk(project_path):
            if "__pycache__" in root or ".git" in root:
                continue
                
            rel_path = os.path.relpath(root, project_path)
            level = len(Path(rel_path).parts)
            indent = "  " * level
            
            if rel_path != ".":
                files_info += f"{indent}📁 {os.path.basename(root)}/\n"
            
            for file in sorted(files):
                if file.startswith(".") or file.endswith(".pyc"):
                    continue
                files_info += f"{indent}  📄 {file}\n"
        
        system_prompt += files_info
        return system_prompt
    
    def handle_tool_calls(self, tool_calls, file_explorer, terminal, project_path, open_files):
        """Handle tool calls from the AI."""
        results = []
        if not tool_calls:
            return results
            
        for tool_call in tool_calls:
            # For Gemini, we'll extract function calls from the response
            # This is a simplified implementation since Gemini doesn't have native function calling like Claude
            try:
                function_name = tool_call.get('name', '')
                args = tool_call.get('args', {})
                
                if function_name == "read_file":
                    try:
                        file_path = os.path.join(project_path, args.get("path", ""))
                        
                        if os.path.exists(file_path):
                            with open(file_path, 'r') as f:
                                content = f.read()
                            results.append({
                                "role": "tool",
                                "name": function_name,
                                "content": f"Content of {args.get('path')}:\n\n```\n{content}\n```"
                            })
                        else:
                            results.append({
                                "role": "tool",
                                "name": function_name,
                                "content": f"Error: File '{args.get('path')}' not found."
                            })
                    except Exception as e:
                        results.append({
                            "role": "tool",
                            "name": function_name,
                            "content": f"Error reading file: {str(e)}"
                        })
                        
                elif function_name == "write_file":
                    try:
                        file_path = os.path.join(project_path, args.get("path", ""))
                        content = args.get("content", "")
                        
                        # Ensure directory exists
                        os.makedirs(os.path.dirname(file_path), exist_ok=True)
                        
                        with open(file_path, 'w') as f:
                            f.write(content)
                        
                        # Update open file if it's currently open
                        if file_path in open_files:
                            open_files[file_path] = content
                        
                        results.append({
                            "role": "tool",
                            "name": function_name,
                            "content": f"File '{args.get('path')}' has been created/updated successfully."
                        })
                    except Exception as e:
                        results.append({
                            "role": "tool",
                            "name": function_name,
                            "content": f"Error writing file: {str(e)}"
                        })
                        
                elif function_name == "execute_command":
                    try:
                        command = args.get("command", "")
                        
                        stdout, stderr, returncode = terminal.execute_command(command)
                        
                        result = f"Command: {command}\n"
                        if stdout:
                            result += f"\nStandard Output:\n{stdout}"
                        if stderr:
                            result += f"\nStandard Error:\n{stderr}"
                        result += f"\nReturn Code: {returncode}"
                        
                        results.append({
                            "role": "tool",
                            "name": function_name,
                            "content": result
                        })
                    except Exception as e:
                        results.append({
                            "role": "tool",
                            "name": function_name,
                            "content": f"Error executing command: {str(e)}"
                        })
                        
                elif function_name == "list_directory":
                    try:
                        dir_path = args.get("path", ".")
                        full_path = os.path.join(project_path, dir_path)
                        
                        if os.path.exists(full_path) and os.path.isdir(full_path):
                            files = os.listdir(full_path)
                            file_info = []
                            
                            for file in sorted(files):
                                file_path = os.path.join(full_path, file)
                                is_dir = os.path.isdir(file_path)
                                file_type = "directory" if is_dir else "file"
                                
                                file_info.append({
                                    "name": file,
                                    "type": file_type,
                                    "size": os.path.getsize(file_path) if not is_dir else None
                                })
                            
                            results.append({
                                "role": "tool",
                                "name": function_name,
                                "content": json.dumps(file_info, indent=2)
                            })
                        else:
                            results.append({
                                "role": "tool",
                                "name": function_name,
                                "content": f"Error: Directory '{dir_path}' not found."
                            })
                    except Exception as e:
                        results.append({
                            "role": "tool",
                            "name": function_name,
                            "content": f"Error listing directory: {str(e)}"
                        })
            except Exception as e:
                results.append({
                    "role": "tool",
                    "name": "error",
                    "content": f"Error processing tool call: {str(e)}"
                })
                    
        return results
    
    def parse_tool_calls(self, response_text):
        """Parse tool calls from the model response."""
        # Simple regex-based parsing for tool calls in the format:
        # ```function_call
        # {
        #   "name": "function_name",
        #   "args": {
        #     "arg1": "value1"
        #   }
        # }
        # ```
        tool_calls = []
        pattern = r"```(?:function_call|json)\s*({[\s\S]*?})\s*```"
        matches = re.findall(pattern, response_text)
        
        for match in matches:
            try:
                function_data = json.loads(match)
                tool_calls.append(function_data)
            except json.JSONDecodeError:
                st.warning(f"Failed to parse function call: {match}")
        
        return tool_calls
    
    def render(self, file_explorer, terminal, open_files, project_path):
        """Render the Cline Assistant interface."""
        # Check if API key is set
        if not st.session_state.client:
            st.warning("Please enter your Google API key in the sidebar to use the Cline Assistant.")
            st.info("After entering your API key, click anywhere outside the text box to apply it.")
            
            # Add instructions for getting an API key
            with st.expander("How to get a Google Gemini API key"):
                st.markdown("""
                1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
                2. Sign in with your Google account
                3. Click on "Get API key" in the top menu
                4. Create a new API key or use an existing one
                5. Copy the API key and paste it in the sidebar
                """)
            return
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            if not st.session_state.chat_history:
                st.markdown("**Cline**: Hello! I'm your AI programming assistant. How can I help you with your coding project today?")
            
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f"**You**: {message['content']}")
                elif message["role"] == "assistant":
                    st.markdown(f"**Cline**: {message['content']}")
                elif message["role"] == "tool_result":
                    with st.expander(f"Tool Result: {message['name']}"):
                        st.code(message['content'], language="plaintext")
        
        # User input
        user_input = st.text_area("Message Cline", height=100, key="user_message")
        submit_button = st.button("Send")
        
        if submit_button and user_input:
            # Add user message to chat history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })
            
            # Clear the input area
            st.session_state.user_message = ""
            
            # Generate system prompt with project context
            system_prompt = self.generate_system_prompt(project_path)
            
            try:
                with st.spinner("Cline is thinking..."):
                    # Create conversation history for Gemini
                    gemini_history = []
                    for message in st.session_state.chat_history:
                        if message["role"] == "user":
                            gemini_history.append({"role": "user", "parts": [message["content"]]})
                        elif message["role"] == "assistant":
                            gemini_history.append({"role": "model", "parts": [message["content"]]})
                    
                    # Initialize Gemini model
                    model = st.session_state.client.GenerativeModel('gemini-pro')
                    
                    # Start the chat
                    chat = model.start_chat(history=gemini_history)
                    
                    # Generate response with system prompt
                    response = chat.send_message(
                        f"{system_prompt}\n\nUser message: {user_input}\n\n"
                        "You can use function calls in this format when needed:\n"
                        "```function_call\n"
                        "{\n"
                        '  "name": "function_name",\n'
                        '  "args": {\n'
                        '    "arg1": "value1"\n'
                        '  }\n'
                        "}\n"
                        "```\n"
                        "Available functions:\n"
                        "- read_file(path): Read content of a file\n"
                        "- write_file(path, content): Write content to a file\n"
                        "- execute_command(command): Execute a command in the terminal\n"
                        "- list_directory(path): List files in a directory"
                    )
                    
                    # Extract the response text
                    assistant_message_content = response.text
                    
                    # Add assistant message to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": assistant_message_content
                    })
                    
                    # Check for tool calls in the response
                    tool_calls = self.parse_tool_calls(assistant_message_content)
                    
                    if tool_calls:
                        # Handle tool calls
                        tool_results = self.handle_tool_calls(
                            tool_calls,
                            file_explorer,
                            terminal,
                            project_path,
                            open_files
                        )
                        
                        for tool_result in tool_results:
                            # Add tool result to chat history
                            st.session_state.chat_history.append({
                                "role": "tool_result",
                                "name": tool_result["name"],
                                "content": tool_result["content"]
                            })
                        
                        # Make a follow-up API call with the tool results
                        with st.spinner("Processing tool results..."):
                            # Format tool results for the model
                            tool_results_text = "\n\n".join([
                                f"Tool: {result['name']}\nResult: {result['content']}"
                                for result in tool_results
                            ])
                            
                            # Send follow-up message with tool results
                            follow_up_response = chat.send_message(
                                f"Here are the results of the function calls:\n\n{tool_results_text}\n\n"
                                "Please provide your response based on these results."
                            )
                            
                            follow_up_content = follow_up_response.text
                            
                            # Add follow-up message to chat history
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": follow_up_content
                            })
                
            except Exception as e:
                st.error(f"Error communicating with Gemini: {str(e)}")
                st.error("Please check your API key and internet connection.")
            
            # Force a rerun to update the chat container
            st.rerun()

# Add additional utility functions
def extract_code_blocks(text):
    """Extract code blocks from markdown text."""
    pattern = r"```(?:\w+)?\s*([\s\S]*?)```"
    return re.findall(pattern, text)

