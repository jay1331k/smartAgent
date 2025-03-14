import streamlit as st
import os
import sys
import subprocess
import tempfile
from pathlib import Path

class Terminal:
    def __init__(self, working_directory=None):
        self.working_directory = working_directory or os.getcwd()
        self.history = []
        
    def run_command(self, command):
        if not command.strip():
            return "No command provided."
        
        try:
            # Create a temporary file to capture output
            with tempfile.TemporaryFile(mode='w+t') as stdout_file, tempfile.TemporaryFile(mode='w+t') as stderr_file:
                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=self.working_directory
                )
                
                stdout, stderr = process.communicate(timeout=15)  # 15 second timeout
                result = f"$ {command}\n"
                if stdout:
                    result += stdout
                if stderr:
                    result += f"\nError:\n{stderr}"
                
                self.history.append({
                    "command": command,
                    "result": result,
                    "exit_code": process.returncode
                })
                return result
                
        except subprocess.TimeoutExpired:
            process.kill()
            return f"Command timed out: {command}"
        except Exception as e:
            return f"Error executing command: {str(e)}"
    
    def get_history(self):
        return self.history
    
    def set_working_directory(self, directory):
        if os.path.isdir(directory):
            self.working_directory = directory
            return True
        return False

class ClineInterface:
    def __init__(self, terminal=None):
        self.terminal = terminal or Terminal()
        
    def display(self):
        # Command input
        command = st.text_input("Enter command:", key="terminal_command")
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("Run"):
                if command:
                    output = self.terminal.run_command(command)
                    if "terminal_history" not in st.session_state:
                        st.session_state.terminal_history = []
                    st.session_state.terminal_history.append(output)
                    st.session_state.terminal_command = ""  # Clear input
        with col2:
            if st.button("Clear History"):
                st.session_state.terminal_history = []
        
        # Display history - FIXED: Using accordion instead of expanders for nesting safety
        if "terminal_history" in st.session_state and st.session_state.terminal_history:
            st.write("**Command History:**")
            for i, entry in enumerate(st.session_state.terminal_history):
                st.write(f"**Command #{i+1}:**")
                st.code(entry, language="bash")
