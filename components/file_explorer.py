import streamlit as st
import os
from pathlib import Path

class FileExplorer:
    def __init__(self, root_path):
        self.root_path = root_path
        # Common file types and their languages for syntax highlighting
        self.file_extensions = {
            '.py': 'python',
            '.js': 'javascript',
            '.html': 'html',
            '.css': 'css',
            '.json': 'json',
            '.md': 'markdown',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.txt': 'text',
            '.csv': 'text',
            '.sh': 'bash',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.rs': 'rust'
        }

    def get_file_tree(self):
        file_tree = []
        for root, dirs, files in os.walk(self.root_path):
            rel_path = os.path.relpath(root, self.root_path)
            depth = rel_path.count(os.sep)
            if rel_path == ".":
                rel_path = ""  # Avoid displaying "." for the root directory
                depth = 0

            # Only add items at the current level
            if depth == 0:
                for dir_name in sorted(dirs):
                    file_tree.append({
                        "name": dir_name,
                        "type": "directory",
                        "path": os.path.join(root, dir_name),
                        "depth": depth
                    })
                for file_name in sorted(files):
                    file_tree.append({
                        "name": file_name,
                        "type": "file",
                        "path": os.path.join(root, file_name),
                        "depth": depth
                    })
        return file_tree

    def display_file_tree(self):
        """Display the file tree without nested expanders"""
        # Get all directories for folder navigation
        directories = []
        for root, dirs, _ in os.walk(self.root_path):
            for dir_name in dirs:
                full_path = os.path.join(root, dir_name)
                rel_path = os.path.relpath(full_path, self.root_path)
                directories.append((rel_path, full_path))
        
        # Show current path
        st.write(f"**Current Directory:** {os.path.basename(self.root_path)}")
        
        # Go up button
        parent_dir = os.path.dirname(self.root_path)
        if st.button("⬆️ Up to Parent Directory"):
            st.session_state.explorer_dir = parent_dir
            st.experimental_rerun()
        
        # List directories first
        st.write("**Folders:**")
        dirs_found = False
        
        for root, dirs, _ in os.walk(self.root_path):
            # Only process immediate subdirectories of the current directory
            if root == self.root_path:
                if not dirs:
                    st.write("*(No folders)*")
                else:
                    dirs_found = True
                    for dir_name in sorted(dirs):
                        dir_path = os.path.join(root, dir_name)
                        if st.button(f"📁 {dir_name}", key=f"dir_{dir_path}"):
                            st.session_state.explorer_dir = dir_path
                            st.experimental_rerun()
            break  # Only process the top level
        
        # List files
        st.write("**Files:**")
        files_found = False
        
        for root, _, files in os.walk(self.root_path):
            # Only process files in the current directory
            if root == self.root_path:
                if not files:
                    st.write("*(No files)*")
                else:
                    files_found = True
                    for file_name in sorted(files):
                        file_path = os.path.join(root, file_name)
                        if st.button(f"📄 {file_name}", key=f"file_{file_path}"):
                            self.display_file_content(file_path)
            break  # Only process the top level
    
    def display_file_content(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                content = file.read()
                st.session_state.current_file = file_path
                st.session_state.file_content = content
                
                # Store file extension for syntax highlighting
                file_ext = os.path.splitext(file_path)[1].lower()
                st.session_state.current_file_language = self.file_extensions.get(file_ext, 'text')
                
                # Signal that a file has been selected
                st.session_state.file_selected = True
        except Exception as e:
            st.error(f"Error opening file: {str(e)}")
            
    def get_file_language(self, file_path):
        file_ext = os.path.splitext(file_path)[1].lower()
        return self.file_extensions.get(file_ext, 'text')
