import streamlit as st
import os
import tempfile
from pathlib import Path
import shutil
from io import StringIO

class FileExplorer:
    def __init__(self, root_path):
        self.root_path = root_path
        
    def get_file_tree(self):
        """Generate a tree structure of files and directories."""
        file_tree = []
        
        for root, dirs, files in os.walk(self.root_path):
            rel_path = os.path.relpath(root, self.root_path)
            if rel_path == ".":
                # Root directory
                for dir_name in sorted(dirs):
                    file_tree.append({
                        "name": dir_name,
                        "type": "directory",
                        "path": os.path.join(self.root_path, dir_name),
                        "children": []
                    })
                
                for file_name in sorted(files):
                    file_tree.append({
                        "name": file_name,
                        "type": "file",
                        "path": os.path.join(self.root_path, file_name),
                        "children": []
                    })
            else:
                # Process subdirectories
                parent_path = os.path.dirname(rel_path)
                dir_name = os.path.basename(rel_path)
                
                # Find parent node
                parent_node = self._find_node(file_tree, parent_path)
                
                if parent_node:
                    # Add current directory
                    dir_node = {
                        "name": dir_name,
                        "type": "directory",
                        "path": os.path.join(self.root_path, rel_path),
                        "children": []
                    }
                    parent_node["children"].append(dir_node)
                    
                    # Add files in this directory
                    for file_name in sorted(files):
                        dir_node["children"].append({
                            "name": file_name,
                            "type": "file",
                            "path": os.path.join(self.root_path, rel_path, file_name),
                            "children": []
                        })
        
        return file_tree
    
    def _find_node(self, tree, path):
        """Find a node in the tree by path."""
        if path == ".":
            return {"children": tree}
        
        parts = path.split(os.path.sep)
        current_tree = tree
        
        for part in parts:
            found = False
            for node in current_tree:
                if node["type"] == "directory" and node["name"] == part:
                    current_tree = node["children"]
                    found = True
                    break
            
            if not found:
                return None
        
        return {"children": current_tree}
    
    def render(self):
        """Render the file explorer in Streamlit."""
        st.write("📁 Project Files")
        
        # File operations
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("New File"):
                st.session_state.new_file_dialog = True
                
        with col2:
            if st.button("New Folder"):
                st.session_state.new_folder_dialog = True
                
        with col3:
            if st.button("Upload File"):
                st.session_state.upload_file_dialog = True
        
        # New file dialog
        if st.session_state.get('new_file_dialog', False):
            with st.form("new_file_form"):
                new_file_name = st.text_input("File name")
                initial_content = st.text_area("Initial content", height=100)
                
                col1, col2 = st.columns(2)
                submit = col1.form_submit_button("Create")
                cancel = col2.form_submit_button("Cancel")
                
                if submit and new_file_name:
                    file_path = os.path.join(self.root_path, new_file_name)
                    try:
                        with open(file_path, 'w') as f:
                            f.write(initial_content)
                        st.success(f"Created file: {new_file_name}")
                        st.session_state.new_file_dialog = False
                        st.rerun()  # Changed from st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error creating file: {str(e)}")
                
                if cancel:
                    st.session_state.new_file_dialog = False
                    st.rerun()  # Changed from st.experimental_rerun()
        
        # New folder dialog
        if st.session_state.get('new_folder_dialog', False):
            with st.form("new_folder_form"):
                new_folder_name = st.text_input("Folder name")
                
                col1, col2 = st.columns(2)
                submit = col1.form_submit_button("Create")
                cancel = col2.form_submit_button("Cancel")
                
                if submit and new_folder_name:
                    folder_path = os.path.join(self.root_path, new_folder_name)
                    try:
                        os.makedirs(folder_path, exist_ok=True)
                        st.success(f"Created folder: {new_folder_name}")
                        st.session_state.new_folder_dialog = False
                        st.rerun()  # Changed from st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error creating folder: {str(e)}")
                
                if cancel:
                    st.session_state.new_folder_dialog = False
                    st.rerun()  # Changed from st.experimental_rerun()
        
        # Upload file dialog
        if st.session_state.get('upload_file_dialog', False):
            uploaded_file = st.file_uploader("Choose a file")
            
            col1, col2 = st.columns(2)
            upload = col1.button("Upload")
            cancel = col2.button("Cancel Upload")
            
            if upload and uploaded_file is not None:
                file_path = os.path.join(self.root_path, uploaded_file.name)
                try:
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.success(f"Uploaded file: {uploaded_file.name}")
                    st.session_state.upload_file_dialog = False
                    st.rerun()  # Changed from st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error uploading file: {str(e)}")
            
            if cancel:
                st.session_state.upload_file_dialog = False
                st.rerun()  # Changed from st.experimental_rerun()
        
        # Display file tree
        try:
            files = os.listdir(self.root_path)
            selected_file = None
            
            for file in sorted(files):
                file_path = os.path.join(self.root_path, file)
                is_dir = os.path.isdir(file_path)
                icon = "📁" if is_dir else "📄"
                
                if is_dir:
                    expander = st.expander(f"{icon} {file}")
                    with expander:
                        selected_subfile = self._render_subdirectory(file_path, "")
                        if selected_subfile:
                            selected_file = selected_subfile
                else:
                    if st.button(f"{icon} {file}", key=f"file_{file}"):
                        selected_file = file_path
            
            return selected_file
            
        except Exception as e:
            st.error(f"Error accessing files: {str(e)}")
            return None
    
    def _render_subdirectory(self, dir_path, indent):
        """Recursively render subdirectories."""
        try:
            files = os.listdir(dir_path)
            selected_file = None
            
            for file in sorted(files):
                file_path = os.path.join(dir_path, file)
                is_dir = os.path.isdir(file_path)
                icon = "📁" if is_dir else "📄"
                
                if is_dir:
                    expander = st.expander(f"{indent}{icon} {file}")
                    with expander:
                        selected_subfile = self._render_subdirectory(file_path, indent + "  ")
                        if selected_subfile:
                            selected_file = selected_subfile
                else:
                    if st.button(f"{indent}{icon} {file}", key=f"file_{file_path}"):
                        selected_file = file_path
            
            return selected_file
            
        except Exception as e:
            st.error(f"Error accessing directory {dir_path}: {str(e)}")
            return None

# Make sure the class is properly exported
if __name__ == "__main__":
    # For testing the file explorer directly
    st.title("File Explorer Test")
    explorer = FileExplorer(tempfile.mkdtemp())
    explorer.render()
