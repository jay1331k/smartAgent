import os
import re
from typing import Dict, List, Optional
import streamlit as st

class FileManager:
    """Handles file operations like creating, updating, and deleting files."""
    
    def __init__(self):
        self.base_path = "e:\\Projects\\smartAgent"
        self.language_map = {
            "py": "python",
            "js": "javascript",
            "ts": "typescript",
            "html": "html",
            "css": "css",
            "json": "json",
            "md": "markdown",
            "txt": "text",
            # Add more mappings as needed
        }
    
    def extract_code_blocks(self, text: str) -> Dict[str, str]:
        """
        Extract code blocks with filepath comments from text.
        Returns a dictionary mapping filepath to code content.
        """
        from components.utils import extract_code_with_filenames
        
        code_files = {}
        extracted = extract_code_with_filenames(text)
        
        # Convert to simple filepath -> content mapping
        for filepath, info in extracted.items():
            code_files[filepath] = info["content"]
                
        return code_files
    
    def save_file(self, filepath: str, content: str) -> bool:
        """Save content to a file, creating directories if needed."""
        try:
            # Ensure the filepath is within the base path
            if not filepath.startswith(self.base_path):
                filepath = os.path.join(self.base_path, filepath.lstrip('/\\'))
            
            # Ensure directory exists
            directory = os.path.dirname(filepath)
            os.makedirs(directory, exist_ok=True)
            
            # Write the file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
                
            return True
        except Exception as e:
            st.error(f"Error saving file {filepath}: {str(e)}")
            return False
    
    def read_file(self, filepath: str) -> Optional[str]:
        """Read content from a file if it exists."""
        try:
            # Ensure the filepath is within the base path
            if not filepath.startswith(self.base_path):
                filepath = os.path.join(self.base_path, filepath.lstrip('/\\'))
                
            if not os.path.exists(filepath):
                return None
                
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            st.error(f"Error reading file {filepath}: {str(e)}")
            return None
    
    def delete_file(self, filepath: str) -> bool:
        """Delete a file if it exists."""
        try:
            # Ensure the filepath is within the base path
            if not filepath.startswith(self.base_path):
                filepath = os.path.join(self.base_path, filepath.lstrip('/\\'))
                
            if not os.path.exists(filepath):
                return False
                
            os.remove(filepath)
            return True
        except Exception as e:
            st.error(f"Error deleting file {filepath}: {str(e)}")
            return False
            
    def get_language_from_extension(self, filepath: str) -> str:
        """Get the language name from file extension for syntax highlighting."""
        ext = os.path.splitext(filepath)[1].lstrip('.')
        return self.language_map.get(ext, "text")

    def extract_and_save_code_from_response(self, node_output: str) -> Dict[str, str]:
        """
        Extract code files from LLM response and save them to disk if they have valid file paths.
        Returns a dictionary mapping successful filepaths to their content.
        """
        saved_files = {}
        code_files = self.extract_code_blocks(node_output)
        
        for filepath, content in code_files.items():
            try:
                # Skip if filepath is invalid or suspicious
                if not filepath or '..' in filepath:
                    continue
                    
                # Ensure the filepath is within the base path
                if not filepath.startswith(self.base_path):
                    filepath = os.path.join(self.base_path, filepath.lstrip('/\\'))
                
                # Ensure directory exists
                directory = os.path.dirname(filepath)
                os.makedirs(directory, exist_ok=True)
                
                # Write file
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
                saved_files[filepath] = content
                st.success(f"File created: {filepath}")
            except Exception as e:
                st.error(f"Error creating file {filepath}: {str(e)}")
        
        return saved_files

    def process_code_result_from_node(self, node) -> None:
        """Process code results from a node's output and provide options to save."""
        code_files = self.extract_code_blocks(node.output)
        
        if code_files:
            st.write("### Code Files Generated")
            st.write(f"Found {len(code_files)} code files in the output.")
            
            # Option to save all files at once
            if len(code_files) > 1:
                if st.button("Save All Files", key=f"save_all_{node.node_id}"):
                    self.extract_and_save_code_from_response(node.output)
            
            # Show each file with option to save individually
            for filepath, content in code_files.items():
                with st.expander(f"File: {filepath}", expanded=False):
                    # Show code with syntax highlighting
                    language = self.get_language_from_extension(filepath)
                    st.code(content, language=language)
                    
                    # Save button for this file
                    if st.button(f"Save {filepath}", key=f"save_{node.node_id}_{filepath}"):
                        success = self.save_file(filepath, content)
                        if success:
                            st.success(f"File saved: {filepath}")
                        else:
                            st.error(f"Failed to save file")
