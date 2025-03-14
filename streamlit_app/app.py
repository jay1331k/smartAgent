import streamlit as st
import os
import sys
import subprocess
import json
import tempfile
from pathlib import Path
import zipfile
import io

# Import custom modules - make sure these imports are correct
from file_explorer import FileExplorer
from terminal import Terminal
from cline_interface import ClineInterface

# Configure page
st.set_page_config(
    page_title="Cline Web IDE",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #1e1e1e;
    }
    .stTabs {
        background-color: #252526;
        border-radius: 5px;
    }
    .stTextInput > div > div > input {
        background-color: #3c3c3c;
        color: #cccccc;
    }
    .css-145kmo2 {
        border: 1px solid #454545;
    }
    .css-1offfwp {
        font-size: 14px;
    }
    .ace_editor {
        min-height: 600px !important;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'project_path' not in st.session_state:
    st.session_state.project_path = tempfile.mkdtemp(prefix="cline_project_")
if 'current_file' not in st.session_state:
    st.session_state.current_file = None
if 'open_files' not in st.session_state:
    st.session_state.open_files = {}
if 'terminal_history' not in st.session_state:
    st.session_state.terminal_history = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'api_key' not in st.session_state:
    st.session_state.api_key = None

# Initialize components
file_explorer = FileExplorer(st.session_state.project_path)
terminal = Terminal(st.session_state.project_path)
cline_interface = ClineInterface()

# Main layout
st.title("Cline Web IDE")

# API key input
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("API Key (Google Gemini)", type="password", key="gemini_api_key")
    if api_key:
        # Only set the API key if it's not already set or if it has changed
        if st.session_state.api_key != api_key:
            st.session_state.api_key = api_key
            if cline_interface.set_api_key(api_key):
                st.success("API key set successfully!")
            else:
                st.error("Failed to initialize the AI with the provided API key")
    
    # Add link to get API key
    st.markdown("[Get a Google Gemini API Key](https://makersuite.google.com/app/apikey)")
    
    st.header("Project")
    if st.button("Download Project"):
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(st.session_state.project_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(
                        file_path, 
                        os.path.relpath(file_path, st.session_state.project_path)
                    )
        
        zip_buffer.seek(0)
        st.download_button(
            label="Download ZIP",
            data=zip_buffer,
            file_name="cline_project.zip",
            mime="application/zip"
        )

# Main content area
col1, col2 = st.columns([1, 3])

# File explorer in the first column
with col1:
    st.subheader("File Explorer")
    selected_file = file_explorer.render()
    if selected_file and selected_file != st.session_state.current_file:
        st.session_state.current_file = selected_file
        if selected_file not in st.session_state.open_files:
            try:
                with open(selected_file, 'r') as f:
                    content = f.read()
                st.session_state.open_files[selected_file] = content
            except Exception as e:
                st.error(f"Error opening file: {str(e)}")

# Editor and terminal in the second column
with col2:
    tabs = st.tabs(["Editor", "Terminal", "Cline Assistant"])
    
    # Editor tab
    with tabs[0]:
        if st.session_state.current_file:
            file_name = os.path.basename(st.session_state.current_file)
            st.subheader(f"Editing: {file_name}")
            
            # Determine file type for syntax highlighting
            file_ext = os.path.splitext(file_name)[1].lower()
            mode = "python"
            if file_ext in ['.js', '.jsx']:
                mode = "javascript"
            elif file_ext in ['.ts', '.tsx']:
                mode = "typescript"
            elif file_ext in ['.html']:
                mode = "html"
            elif file_ext in ['.css']:
                mode = "css"
            elif file_ext in ['.json']:
                mode = "json"
            elif file_ext in ['.md']:
                mode = "markdown"
            
            # Use streamlit-ace for code editing
            from streamlit_ace import st_ace
            
            content = st.session_state.open_files[st.session_state.current_file]
            new_content = st_ace(
                value=content,
                language=mode,
                theme="monokai",
                key=st.session_state.current_file,
                auto_update=True
            )
            
            if new_content != content:
                st.session_state.open_files[st.session_state.current_file] = new_content
                try:
                    with open(st.session_state.current_file, 'w') as f:
                        f.write(new_content)
                    st.success("File saved successfully!")
                except Exception as e:
                    st.error(f"Error saving file: {str(e)}")
        else:
            st.info("Select a file from the explorer to edit")

    # Terminal tab
    with tabs[1]:
        st.subheader("Terminal")
        terminal_output = terminal.render()
        if terminal_output:
            st.session_state.terminal_history.append(terminal_output)
        
    # Cline Assistant tab
    with tabs[2]:
        st.subheader("Cline Assistant")
        cline_interface.render(
            file_explorer=file_explorer,
            terminal=terminal,
            open_files=st.session_state.open_files,
            project_path=st.session_state.project_path
        )

# Footer
st.markdown("---")
st.markdown("Cline Web IDE - Powered by Streamlit")
