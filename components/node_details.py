import streamlit as st
import json
import os
from typing import Optional, Dict, Any

from components.utils import STATUS_PENDING, STATUS_RUNNING, STATUS_COMPLETED, STATUS_FAILED

def display_node_details(node_id: str):
    """Display detailed information about a selected node"""
    if node_id not in st.session_state.node_lookup:
        st.warning("Selected node not found.")
        return
        
    node = st.session_state.node_lookup[node_id]
    
    # Header with status indicator
    status_emoji = {
        STATUS_PENDING: "🔘",
        STATUS_RUNNING: "⏳",
        STATUS_COMPLETED: "✅",
        STATUS_FAILED: "❌"
    }.get(node.status, "⚪")
    
    st.subheader(f"{status_emoji} {node.retrieve_from_memory('task')}")
    
    # Basic information
    st.write(f"**Status:** {node.status}")
    st.write(f"**Depth:** {node.depth}")
    st.write(f"**ID:** {node.node_id}")
    
    # Show parent information if available
    if node.parent_id and node.parent_id in st.session_state.node_lookup:
        parent = st.session_state.node_lookup[node.parent_id]
        parent_task = parent.retrieve_from_memory("task")
        st.write(f"**Parent:** {parent_task}")
        if st.button("Go to parent", key=f"goto_parent_{node.node_id}"):
            st.session_state.selected_node_id = node.parent_id
            st.experimental_rerun()
    
    # Actions based on node status
    st.write("### Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if node.status == STATUS_PENDING:
            if st.button("Execute", key=f"execute_{node.node_id}"):
                # Execute in a way that avoids nested expanders
                with st.spinner("Executing..."):
                    st.session_state.agent.agentFlow("execute", node)
                st.experimental_rerun()
                
    with col2:
        if node.status in [STATUS_COMPLETED, STATUS_FAILED]:
            regeneration_guidance = st.text_area(
                "Guidance for regeneration:", 
                key=f"guidance_{node.node_id}",
                placeholder="Provide instructions for regeneration..."
            )
            if st.button("Regenerate", key=f"regenerate_{node.node_id}"):
                with st.spinner("Regenerating..."):
                    st.session_state.agent.agentFlow("regenerate", node, regeneration_guidance)
                st.experimental_rerun()
    
    with col3:
        if st.button("Delete", key=f"delete_{node.node_id}"):
            # Special handling for root node
            if node.node_id == st.session_state.root_node_id:
                st.session_state.root_node_id = None
            
            st.session_state.agent.agentFlow("delete", node)
            st.session_state.selected_node_id = None
            st.experimental_rerun()
    
    # Output section - Use tabs instead of nested expanders
    if node.output:
        tabs = st.tabs(["Output", "Generated Code"])
        
        # Output tab
        with tabs[0]:
            st.code(node.output, language="text")
        
        # Code files tab
        with tabs[1]:
            code_files = node.extract_code_files()
            if code_files:
                st.write("### Generated Code Files")
                
                # Use a selectbox for file selection
                file_paths = list(code_files.keys())
                selected_file = st.selectbox("Select a file to view:", file_paths, key=f"file_select_{node.node_id}")
                
                if selected_file:
                    content = code_files[selected_file]
                    ext = os.path.splitext(selected_file)[1][1:] if '.' in selected_file else ''
                    lang = {
                        'py': 'python',
                        'js': 'javascript',
                        'html': 'html',
                        'css': 'css',
                        'json': 'json',
                        'md': 'markdown'
                    }.get(ext, 'text')
                    
                    st.code(content, language=lang)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"Save file", key=f"save_file_{node.node_id}"):
                            try:
                                directory = os.path.dirname(selected_file)
                                if directory and not os.path.exists(directory):
                                    os.makedirs(directory, exist_ok=True)
                                    
                                with open(selected_file, 'w', encoding='utf-8') as f:
                                    f.write(content)
                                    
                                st.success(f"File saved: {selected_file}")
                            except Exception as e:
                                st.error(f"Error saving file: {str(e)}")
                    
                    with col2:
                        if st.button(f"Open in editor", key=f"open_editor_{node.node_id}"):
                            st.session_state.active_file = selected_file
                            st.session_state.file_content = content
                            st.experimental_rerun()
            else:
                st.write("No code files generated.")
                
    
    # Error message if any
    if node.error_message:
        st.error(f"Error: {node.error_message}")
    
    # Add child task section
    if node.status == STATUS_COMPLETED:
        st.write("### Add Child Task")
        
        new_task = st.text_area(
            "New task description:",
            key=f"new_task_{node.node_id}",
            placeholder="Describe a subtask to add..."
        )
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Add Subtask", key=f"add_subtask_{node.node_id}"):
                if new_task.strip():
                    # Create child node
                    child_node = st.session_state.agent.create_child_node(
                        node,
                        new_task,
                        node.depth + 1
                    )
                    # Select the new node
                    st.session_state.selected_node_id = child_node.node_id
                    st.experimental_rerun()
                else:
                    st.error("Please enter a task description.")
    
    # Child nodes section
    if node.child_ids:
        st.write("### Child Tasks")
        
        # Group children by status
        status_groups = {
            STATUS_PENDING: [],
            STATUS_RUNNING: [],
            STATUS_COMPLETED: [],
            STATUS_FAILED: []
        }
        
        for child_id in node.child_ids:
            if child_id in st.session_state.node_lookup:
                child = st.session_state.node_lookup[child_id]
                status_groups[child.status].append(child)
        
        # Display groups with their status indicators
        for status, emoji in [
            (STATUS_RUNNING, "⏳"),
            (STATUS_PENDING, "🔘"),
            (STATUS_COMPLETED, "✅"),
            (STATUS_FAILED, "❌")
        ]:
            nodes = status_groups[status]
            if nodes:
                st.write(f"**{emoji} {status.capitalize()} ({len(nodes)}):**")
                
                for child in nodes:
                    task = child.retrieve_from_memory("task")
                    if st.button(f"{task}", key=f"child_{child.node_id}"):
                        st.session_state.selected_node_id = child.node_id
                        st.experimental_rerun()
