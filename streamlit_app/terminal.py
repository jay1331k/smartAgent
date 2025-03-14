import streamlit as st
import subprocess
import os
import platform
import time

class Terminal:
    def __init__(self, cwd):
        self.cwd = cwd
        if 'terminal_output' not in st.session_state:
            st.session_state.terminal_output = ["Welcome to Cline Terminal. Type commands below."]
        if 'command_history' not in st.session_state:
            st.session_state.command_history = []
    
    def execute_command(self, command):
        """Execute a command in the terminal and return output."""
        try:
            # Add command to output
            st.session_state.terminal_output.append(f"$ {command}")
            st.session_state.command_history.append(command)
            
            # Execute command
            if platform.system() == "Windows":
                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=self.cwd
                )
            else:
                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=self.cwd
                )
            
            stdout, stderr = process.communicate()
            
            # Add output to terminal
            if stdout:
                st.session_state.terminal_output.append(stdout)
            if stderr:
                st.session_state.terminal_output.append(stderr)
            
            # Keep only the last 100 lines
            if len(st.session_state.terminal_output) > 100:
                st.session_state.terminal_output = st.session_state.terminal_output[-100:]
            
            return stdout, stderr, process.returncode
            
        except Exception as e:
            error_msg = f"Error executing command: {str(e)}"
            st.session_state.terminal_output.append(error_msg)
            return "", error_msg, 1
    
    def render(self):
        """Render the terminal interface without using streamlit-terminal package."""
        # Display terminal output with custom styling
        st.markdown("""
        <style>
        .terminal-container {
            background-color: #1e1e1e;
            color: #dcdcdc;
            font-family: 'Courier New', monospace;
            padding: 10px;
            border-radius: 5px;
            height: 300px;
            overflow-y: auto;
            white-space: pre-wrap;
            word-break: break-word;
        }
        .terminal-line {
            margin: 0;
            padding: 2px 0;
            line-height: 1.3;
        }
        .terminal-prompt {
            color: #4ec9b0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Display terminal output using custom HTML
        terminal_text = ""
        for line in st.session_state.terminal_output:
            if line.startswith("$ "):
                terminal_text += f'<div class="terminal-line"><span class="terminal-prompt">{line}</span></div>'
            else:
                terminal_text += f'<div class="terminal-line">{line}</div>'
                
        st.markdown(f'<div class="terminal-container">{terminal_text}</div>', unsafe_allow_html=True)
        
        # Command input
        command = st.text_input("Enter command", key="terminal_command")
        col1, col2 = st.columns([1, 5])
        
        if col1.button("Run") or (command and st.session_state.get('enter_pressed', False)):
            if command:
                # Reset enter_pressed flag
                st.session_state.enter_pressed = False
                
                # Execute command
                stdout, stderr, returncode = self.execute_command(command)
                
                # Clear input
                st.session_state.terminal_command = ""
                
                # Return output for potential use by other components
                return {
                    "command": command,
                    "stdout": stdout,
                    "stderr": stderr,
                    "returncode": returncode,
                    "timestamp": time.time()
                }
        
        # Add JavaScript to allow Enter key to execute commands
        st.markdown("""
        <script>
        document.addEventListener('DOMContentLoaded', (event) => {
            const inputField = document.querySelector('input[data-testid="stTextInput"][aria-label="Enter command"]');
            if (inputField) {
                inputField.addEventListener('keydown', function(e) {
                    if (e.key === 'Enter') {
                        const buttons = document.querySelectorAll('button');
                        for (const button of buttons) {
                            if (button.innerText.includes('Run')) {
                                button.click();
                                break;
                            }
                        }
                    }
                });
            }
        });
        </script>
        """, unsafe_allow_html=True)
        
        # Command history dropdown
        if st.session_state.command_history:
            selected_history = col2.selectbox(
                "History", 
                options=[""] + list(reversed(st.session_state.command_history)),
                label_visibility="collapsed"
            )
            if selected_history:
                st.session_state.terminal_command = selected_history
                st.rerun()  # Changed from st.experimental_rerun()
        
        return None
