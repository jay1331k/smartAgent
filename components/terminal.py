import streamlit as st
import os
import sys
import subprocess
import tempfile
from pathlib import Path
import shlex

class Terminal:
    def __init__(self, working_directory=None):
        self.working_directory = working_directory or os.getcwd()
        self.history = []
        
        # Set environment variables for the terminal
        self.env = os.environ.copy()
        # Store the operating system
        self.windows = os.name == 'nt'
        
        # Add Python/common executable paths to PATH
        if self.windows:  # Windows-specific paths
            python_location = sys.executable
            python_dir = os.path.dirname(python_location)
            
            # Add Python installation paths
            if python_dir not in self.env['PATH']:
                self.env['PATH'] = f"{python_dir};{python_dir}\\Scripts;" + self.env['PATH']
                
            # Add common Windows tool locations - especially important for commands like ls, git, etc.
            common_paths = [
                "C:\\Program Files\\Git\\bin",
                "C:\\Program Files\\Git\\cmd",
                "C:\\Program Files\\Git\\usr\\bin",  # For ls, grep, etc
                "C:\\msys64\\usr\\bin",             # MSYS2 paths
                "C:\\msys64\\mingw64\\bin",
                "C:\\Windows\\System32",
                "C:\\Windows",
            ]
            
            for path in common_paths:
                if os.path.exists(path) and path not in self.env['PATH']:
                    self.env['PATH'] += f";{path}"
            
            self.shell = True
            
            # Common command aliases for Windows
            self.command_aliases = {
                'ls': 'dir',
                'cat': 'type',
                'cp': 'copy',
                'mv': 'move',
                'rm': 'del',
                'grep': 'findstr',
                'pwd': 'cd'
            }
        else:  # Unix paths
            python_location = sys.executable
            python_dir = os.path.dirname(python_location)
            if python_dir not in self.env['PATH']:
                self.env['PATH'] = f"{python_dir}:{self.env['PATH']}"
                
            common_paths = [
                "/usr/local/bin",
                "/usr/bin",
                "/bin",
                "/usr/sbin",
                "/sbin",
            ]
            for path in common_paths:
                if os.path.exists(path) and path not in self.env['PATH']:
                    self.env['PATH'] += f":{path}"
            
            self.shell = True
            self.command_aliases = {}  # No need for aliases on Unix
        
    def run_command(self, command):
        if not command.strip():
            return "No command provided."
        
        # Handle commonly mistyped commands
        cmd_lower = command.strip().lower()
        
        # Handle pip --version (note common mistake: -version instead of --version)
        if cmd_lower in ['pip -version', 'pip version', 'pip -v']:
            return self._run_pip_command('pip --version')
        
        try:
            # Special commands handling
            if command.strip().lower() == 'clear':
                self.history = []
                return "Terminal cleared."
                
            # Handle cd command specially because it affects working directory
            if command.strip().startswith('cd '):
                return self._change_directory(command.strip()[3:])
                
            # Handle pwd command on Windows
            if command.strip().lower() == 'pwd':
                return f"$ pwd\n{self.working_directory}"
                
            # Special handling for pip command to ensure we find it
            if command.strip().lower().startswith('pip '):
                return self._run_pip_command(command)

            # Handle aliased commands on Windows
            cmd_parts = command.strip().split(None, 1)
            base_cmd = cmd_parts[0].lower()
            
            if self.windows and base_cmd in self.command_aliases:
                windows_cmd = self.command_aliases[base_cmd]
                args = cmd_parts[1] if len(cmd_parts) > 1 else ""
                command = f"{windows_cmd} {args}"
            
            # Execute in the working directory
            process = subprocess.Popen(
                command,
                shell=self.shell,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.working_directory,
                env=self.env
            )
            
            stdout, stderr = process.communicate(timeout=15)  # 15 second timeout
            result = f"$ {command}\n"
            if stdout:
                result += stdout
            if stderr:
                result += f"\nError:\n{stderr}"
            
            self.history.append(result)
            return result
                
        except subprocess.TimeoutExpired:
            process.kill()
            return f"$ {command}\nCommand timed out after 15 seconds"
        except Exception as e:
            return f"$ {command}\nError executing command: {str(e)}"
    
    def _run_pip_command(self, command):
        """Special handling for pip commands since they often have path issues"""
        # Get Python executable path
        python_exe = sys.executable
        
        # Replace pip with python -m pip
        pip_cmd = command.replace('pip', f'"{python_exe}" -m pip', 1)
        
        try:
            process = subprocess.Popen(
                pip_cmd,
                shell=self.shell,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.working_directory,
                env=self.env
            )
            
            stdout, stderr = process.communicate(timeout=30)  # pip commands may take longer
            result = f"$ {command}\n"
            if stdout:
                result += stdout
            if stderr:
                result += f"\nError:\n{stderr}"
            
            return result
        except Exception as e:
            return f"$ {command}\nError executing pip: {str(e)}"
    
    def _change_directory(self, path):
        """Handle cd command by changing working directory"""
        # Handle home directory alias
        if path.startswith('~'):
            path = os.path.expanduser(path)
            
        # Handle relative paths
        if not os.path.isabs(path):
            target_dir = os.path.join(self.working_directory, path)
        else:
            target_dir = path
            
        # Normalize path
        target_dir = os.path.normpath(target_dir)
        
        if os.path.exists(target_dir) and os.path.isdir(target_dir):
            self.working_directory = target_dir
            return f"$ cd {path}\nChanged directory to: {target_dir}"
        else:
            return f"$ cd {path}\nError: The system cannot find the path specified."
    
    def get_history(self):
        return self.history
    
    def set_working_directory(self, directory):
        if os.path.isdir(directory):
            self.working_directory = directory
            return True
        return False
    
    def get_working_directory(self):
        return self.working_directory

class ClineInterface:
    def __init__(self, terminal=None):
        self.terminal = terminal or Terminal()
        self.initialize_terminal()
        
    def initialize_terminal(self):
        """Set the terminal to the current explorer directory when available"""
        if 'explorer_dir' in st.session_state and os.path.isdir(st.session_state.explorer_dir):
            self.terminal.set_working_directory(st.session_state.explorer_dir)
        
    def display(self):
        # Show current directory
        st.write(f"📂 **Current directory:** `{self.terminal.get_working_directory()}`")
        
        with st.expander("Terminal Help", expanded=False):
            st.markdown("""
            ### Terminal Help
            - Windows users: Both Windows commands and some Unix-like commands are supported
            - For Python/pip commands, if you have issues, try: `python -m pip` instead of just `pip`
            - Use `cd` to change directories, `pwd` to show current directory
            - Use `clear` to clear terminal history
            """)
        
        # Initialize the command in session state if needed
        if 'terminal_command' not in st.session_state:
            st.session_state.terminal_command = ""
        
        # Check if we should submit a command - this is set by the run button
        if 'run_cmd_clicked' not in st.session_state:
            st.session_state.run_cmd_clicked = False
            
        # Store current command input separately to avoid session state issues
        if 'current_cmd_input' not in st.session_state:
            st.session_state.current_cmd_input = ""
        
        # Create a callback function to handle the run button click
        def on_run_click():
            # Set the flag but don't modify session state variables yet
            st.session_state.run_cmd_clicked = True
            st.session_state.current_cmd_input = st.session_state.terminal_command
        
        # Command input with a more compact design
        cmd_col1, cmd_col2 = st.columns([4, 1])
        
        with cmd_col1:
            # Use a text input for the command, but don't modify its state directly
            st.text_input("", 
                         key="terminal_command",
                         placeholder="Enter command here...",
                         label_visibility="collapsed")
        with cmd_col2:
            # Button to submit command
            st.button("Run", key="run_terminal_btn", on_click=on_run_click, use_container_width=True)
        
        # Now check if the button was clicked and handle command execution
        if st.session_state.run_cmd_clicked and st.session_state.current_cmd_input:
            command = st.session_state.current_cmd_input
            output = self.terminal.run_command(command)
            
            # Add to history
            if "terminal_history" not in st.session_state:
                st.session_state.terminal_history = []
            st.session_state.terminal_history.append(output)
            
            # Reset the flag and command input for the next run
            st.session_state.run_cmd_clicked = False
            st.session_state.current_cmd_input = ""
            
            # We need to rerun to update the UI with the new output and clear the field
            st.rerun()
        
        # Terminal controls
        control_col1, control_col2, control_col3 = st.columns([1, 1, 4])
        with control_col1:
            if st.button("Clear", key="clear_terminal"):
                st.session_state.terminal_history = []
                st.experimental_rerun()
        
        with control_col2:
            if st.button("Sync Dir", key="sync_dir", help="Sync terminal with file explorer directory"):
                if 'explorer_dir' in st.session_state:
                    self.terminal.set_working_directory(st.session_state.explorer_dir)
                    st.success(f"Terminal directory synced to: {st.session_state.explorer_dir}")
                    st.experimental_rerun()
        
        # Display history in reverse chronological order (newest at top like VS Code)
        if "terminal_history" in st.session_state and st.session_state.terminal_history:
            # Use a scrollable container with fixed height for terminal output
            st.markdown("""
            <style>
            .terminal-output {
                background-color: #1e1e1e;
                color: #dcdcdc;
                font-family: 'Courier New', monospace;
                padding: 10px;
                height: auto;
                max-height: 200px;
                overflow-y: auto;
                margin-top: 10px;
                border-radius: 4px;
                white-space: pre-wrap;
                word-wrap: break-word;
            }
            .terminal-line {
                padding: 2px 0;
            }
            .command-line {
                color: #569cd6;
                font-weight: bold;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Terminal output
            output_html = "<div class='terminal-output'>"
            for entry in reversed(st.session_state.terminal_history):
                lines = entry.split('\n')
                for i, line in enumerate(lines):
                    if i == 0 and line.startswith('$'):  # Command line
                        output_html += f"<div class='terminal-line command-line'>{line}</div>"
                    else:
                        output_html += f"<div class='terminal-line'>{line}</div>"
            output_html += "</div>"
            
            st.markdown(output_html, unsafe_allow_html=True)
