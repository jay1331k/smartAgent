import streamlit as st
import os
import subprocess
import tempfile
from pathlib import Path
import time
from typing import Dict, List, Optional, Any, Tuple

from streamlit_agraph import agraph, Node as AGraphNode, Edge, Config

# File Explorer Component
class FileExplorer:
    def __init__(self, root_path):
        self.root_path = root_path
        
    def render(self):
        st.write("### Files")
        
        # Create new file/folder section
        with st.expander("Create New"):
            col1, col2 = st.columns(2)
            with col1:
                new_file_name = st.text_input("New File Name", key="new_file_name")
                if st.button("Create File"):
                    if new_file_name:
                        new_path = os.path.join(self.root_path, new_file_name)
                        try:
                            with open(new_path, 'w') as f:
                                pass  # Create empty file
                            st.success(f"Created {new_file_name}")
                            return new_path
                        except Exception as e:
                            st.error(f"Error creating file: {str(e)}")
            
            with col2:
                new_folder_name = st.text_input("New Folder Name", key="new_folder_name")
                if st.button("Create Folder"):
                    if new_folder_name:
                        new_path = os.path.join(self.root_path, new_folder_name)
                        try:
                            os.makedirs(new_path, exist_ok=True)
                            st.success(f"Created {new_folder_name}/")
                        except Exception as e:
                            st.error(f"Error creating folder: {str(e)}")
        
        # Upload file section
        with st.expander("Upload File"):
            uploaded_file = st.file_uploader("Choose a file")
            if uploaded_file:
                file_path = os.path.join(self.root_path, uploaded_file.name)
                try:
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.success(f"Saved {uploaded_file.name}")
                    return file_path
                except Exception as e:
                    st.error(f"Error saving file: {str(e)}")
        
        # Display file tree
        return self._display_files_recursive(self.root_path)
    
    def _display_files_recursive(self, directory, prefix="", is_last=True, depth=0):
        # Limit recursion depth for performance
        if depth > 3:
            st.write(f"{prefix}📂 ... (more files)")
            return None
            
        # List all files and directories
        try:
            items = sorted(os.listdir(directory))
        except PermissionError:
            st.write(f"{prefix}📁 Permission denied")
            return None
        except FileNotFoundError:
            st.write(f"{prefix}❌ Directory not found")
            return None
            
        selected_file = None
        
        # Display folders first, then files
        folders = [item for item in items if os.path.isdir(os.path.join(directory, item))]
        files = [item for item in items if os.path.isfile(os.path.join(directory, item))]
        
        # Display folders
        for i, folder in enumerate(folders):
            is_last_item = (i == len(folders) - 1 and len(files) == 0)
            folder_path = os.path.join(directory, folder)
            
            # Create expandable section for folder
            with st.expander(f"📁 {folder}"):
                result = self._display_files_recursive(folder_path, 
                                                     prefix + ("└── " if is_last_item else "├── "),
                                                     is_last_item,
                                                     depth + 1)
                if result:
                    selected_file = result
        
        # Display files
        for i, file in enumerate(files):
            is_last_item = (i == len(files) - 1)
            file_path = os.path.join(directory, file)
            
            # Create a button for each file
            if st.button(f"📄 {file}", key=f"file_{file_path}"):
                return file_path
                
        return selected_file

# Terminal Component
class Terminal:
    def __init__(self, working_directory):
        self.working_directory = working_directory
        
    def render(self):
        command = st.text_input("Enter command:", key="terminal_cmd")
        if st.button("Execute", key="exec_cmd"):
            if command:
                return self.execute_command(command)
        return None
    
    def execute_command(self, command):
        # Execute the command in the working directory
        try:
            process = subprocess.Popen(
                command,
                shell=True,
                cwd=self.working_directory,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(timeout=30)
            return_code = process.returncode
            
            result = f"$ {command}\n"
            if stdout:
                result += stdout
            if stderr:
                result += f"\nError: {stderr}"
                
            st.code(result)
            return stdout, stderr, return_code
            
        except subprocess.TimeoutExpired:
            process.kill()
            st.error(f"Command timed out: {command}")
            return "", f"Command timed out after 30 seconds", 1
        except Exception as e:
            st.error(f"Error executing command: {str(e)}")
            return "", str(e), 1
        
        return None

# Task Hierarchy Display Component
def render_task_hierarchy(node_lookup, selected_node_id=None):
    if not node_lookup:
        return None
        
    try:
        nodes = []
        edges = []
        
        for node_id, node in node_lookup.items():
            # Node styling based on status
            if node.status == "completed":
                color = "#a8f0b8"  # Light green
            elif node.status == "running":
                color = "#a8d0f0"  # Light blue
            elif node.status == "failed":
                color = "#f0a8a8"  # Light red
            elif node.status == "pending":
                color = "#f0b8a8"  # Light purple
            else:
                color = "#f0d8a8"  # Light orange

            # Create node
            node_style = {
                "size": 25 if node_id != selected_node_id else 35,
                "shape": "circularImage",
                "color": color if node_id != selected_node_id else "#ff0000",
                "image": "https://cdn-icons-png.flaticon.com/512/8859/8859891.png",
                "label": f"{node.node_id[:8]}...",
                "title": node.task_description if hasattr(node, "task_description") else 
                         node.retrieve_from_memory("task"),
            }
            
            nodes.append(AGraphNode(id=node_id, **node_style))
            
            # Create edges
            if node.parent_id:
                edges.append(Edge(source=node.parent_id, target=node_id, type="CURVE_SMOOTH"))
        
        # Configure graph
        config = Config(
            width=750,
            height=500,
            directed=True,
            physics=False,  # Disable physics for hierarchical layout
            hierarchical=True,
            nodeHighlightBehavior=True,
            highlightColor="#f0f0f0",
        )
        
        # Render graph
        return agraph(nodes=nodes, edges=edges, config=config)
        
    except Exception as e:
        st.error(f"Error generating task hierarchy: {str(e)}")
        return None
