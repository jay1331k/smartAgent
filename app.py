"""
Main entry point for the SmartAgent application.
This provides a basic Streamlit interface for interacting with the agent.
"""
import os
import streamlit as st
import uuid
from typing import Dict, Any, List, Optional

from agent.node import Node
from agent.memory import GlobalMemory
from agent.memory_utils import parse_response, safe_serialize

# Initialize global memory
global_memory = GlobalMemory()
nodes: Dict[str, Node] = {}
current_node_id: Optional[str] = None

def initialize_agent():
    """Initialize the agent and create root node if needed."""
    global global_memory, nodes, current_node_id
    
    # Load global memory if it exists
    global_memory.load_from_disk()
    
    # Create a root node if none exists
    if not nodes:
        root_node = Node(node_id="root", node_type="root")
        root_node.store_in_memory("task", "Root task")
        root_node.update_status("completed")
        nodes["root"] = root_node
        current_node_id = "root"

def create_task_node(task_description: str, parent_id: str = "root") -> str:
    """
    Create a new task node.
    
    Args:
        task_description: Description of the task
        parent_id: ID of parent node
    
    Returns:
        ID of the new node
    """
    global nodes
    
    node_id = str(uuid.uuid4())
    new_node = Node(node_id=node_id, node_type="task")
    new_node.set_parent(parent_id)
    new_node.store_in_memory("task", task_description)
    new_node.update_status("pending")
    
    # Update parent node
    if parent_id in nodes:
        nodes[parent_id].add_child(node_id)
    
    # Store the new node
    nodes[node_id] = new_node
    return node_id

def process_task(node_id: str, llm_response: str) -> None:
    """
    Process an LLM response for a task.
    
    Args:
        node_id: ID of the node
        llm_response: Raw response from LLM
    """
    if node_id not in nodes:
        return
    
    node = nodes[node_id]
    node.store_llm_response(llm_response)
    
    # Check if response contains subtasks
    parsed = parse_response(llm_response)
    if isinstance(parsed, dict) and "subtasks" in parsed:
        subtasks = parsed["subtasks"]
        if isinstance(subtasks, list):
            for subtask in subtasks:
                if isinstance(subtask, dict) and "task_description" in subtask:
                    task_desc = subtask["task_description"]
                    create_task_node(task_desc, node_id)
    
    # Update node status
    node.update_status("completed")

def save_agent_state():
    """Save all agent state to disk."""
    global global_memory, nodes
    
    # Save global memory
    global_memory.save_to_disk()
    
    # Save all nodes
    for node_id, node in nodes.items():
        node.save_state()

def load_agent_state():
    """Load all agent state from disk."""
    global global_memory, nodes, current_node_id
    
    # Load global memory
    global_memory.load_from_disk()
    
    # Check for node memory directory
    if os.path.exists("node_memory"):
        # Get all node files
        for filename in os.listdir("node_memory"):
            if filename.endswith(".json"):
                node_id = filename.replace(".json", "")
                new_node = Node(node_id=node_id)
                if new_node.load_state():
                    nodes[node_id] = new_node
                    
        # If we loaded nodes but no current node is set
        if nodes and current_node_id is None:
            current_node_id = "root" if "root" in nodes else list(nodes.keys())[0]

def render_node_tree(root_id: str = "root", level: int = 0):
    """
    Render a node tree in Streamlit.
    
    Args:
        root_id: ID of the root node to start from
        level: Current indentation level
    """
    if root_id not in nodes:
        return
    
    node = nodes[root_id]
    task = node.retrieve_from_memory("task") or "No task"
    status = node.status
    
    # Render this node with proper indentation
    prefix = "  " * level
    label = f"{prefix}•  {task} ({status})"
    
    if st.button(label, key=f"node_{root_id}"):
        st.session_state.selected_node = root_id
    
    # Render children
    for child_id in node.children:
        render_node_tree(child_id, level + 1)

def render_node_details():
    """Render details for the currently selected node."""
    if "selected_node" not in st.session_state or st.session_state.selected_node not in nodes:
        return
    
    node_id = st.session_state.selected_node
    node = nodes[node_id]
    
    st.subheader("Node Details")
    
    # Display node info
    task = node.retrieve_from_memory("task") or "No task"
    status = node.status
    
    st.write(f"**Task:** {task}")
    st.write(f"**Status:** {status}")
    
    # If the node has an LLM response, show it
    raw_response = node.retrieve_from_memory("raw_llm_response")
    if raw_response:
        with st.expander("LLM Response"):
            st.write(raw_response)
    
    # If the node has subtasks, show them
    if node.children:
        st.write("**Subtasks:**")
        for child_id in node.children:
            if child_id in nodes:
                child_task = nodes[child_id].retrieve_from_memory("task") or "No task"
                child_status = nodes[child_id].status
                st.write(f"- {child_task} ({child_status})")

def main():
    """Main application entry point."""
    st.set_page_config(page_title="SmartAgent", layout="wide")
    
    # Initialize session state
    if "selected_node" not in st.session_state:
        st.session_state.selected_node = None
    
    # Initialize agent
    initialize_agent()
    
    # Add a title
    st.title("SmartAgent")
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Tasks")
        
        # Show node tree
        render_node_tree()
        
        # Add new task form
        with st.form("new_task_form"):
            st.write("Create new task")
            task_input = st.text_input("Task description")
            submitted = st.form_submit_button("Create Task")
            
            if submitted and task_input:
                create_task_node(task_input)
                st.experimental_rerun()
    
    with col2:
        # Show node details
        render_node_details()
        
        # If a node is selected, show form to add LLM response
        if "selected_node" in st.session_state and st.session_state.selected_node:
            with st.form("llm_response_form"):
                st.write("Add LLM response")
                response_input = st.text_area("LLM response", height=200)
                submitted = st.form_submit_button("Process Response")
                
                if submitted and response_input:
                    process_task(st.session_state.selected_node, response_input)
                    st.experimental_rerun()
    
    # Add save/load buttons
    if st.button("Save Agent State"):
        save_agent_state()
        st.success("Agent state saved!")
    
    if st.button("Load Agent State"):
        load_agent_state()
        st.success("Agent state loaded!")
        st.experimental_rerun()

if __name__ == "__main__":
    main()
