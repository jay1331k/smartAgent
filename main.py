import streamlit as st
import os
import json
import time
from typing import Optional, Dict, Any

# Import Google's Generative AI library
import google.generativeai as genai
import matplotlib.pyplot as plt
import networkx as nx

# Import SmartAgent components
from components.agent import Agent
from components.node import Node
from components.graph_view import GraphView
from components.utils import (
    initialize_gemini_api, 
    get_model, 
    STATUS_PENDING, 
    STATUS_RUNNING, 
    STATUS_COMPLETED, 
    STATUS_FAILED
)
from components.file_manager import FileManager

# Configure page
st.set_page_config(
    page_title="SmartAgent - Hierarchical Task Decomposition",
    page_icon="🤖",
    layout="wide"
)

# Setup styling
st.markdown("""
<style>
    .task-node {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .task-node-pending {
        background-color: #f0f0f0;
    }
    .task-node-running {
        background-color: #fff3cd;
    }
    .task-node-completed {
        background-color: #d4edda;
    }
    .task-node-failed {
        background-color: #f8d7da;
    }
    .node-controls {
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.node_lookup = {}
    st.session_state.root_node_id = None
    st.session_state.selected_node_id = None
    st.session_state.previous_state = None

# Initialize API and model
if not initialize_gemini_api():
    st.error("Failed to initialize Gemini API. Please check your API key.")
    st.stop()

model = get_model()
if not model:
    st.error("Failed to initialize model. Please check your API key and model availability.")
    st.stop()

# Initialize agent if not already done
if "agent" not in st.session_state:
    generation_config = {
        "temperature": 0.2,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
    }
    st.session_state.agent = Agent(model, generation_config)
    st.session_state.initialized = True
    st.session_state.graph_view = GraphView()
    st.session_state.file_manager = FileManager()

# Layout with sidebar
with st.sidebar:
    st.title("SmartAgent 🤖")
    st.write("Hierarchical Task Decomposition")

    # Model settings
    with st.expander("Model Settings", expanded=False):
        temperature = st.slider(
            "Temperature:",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.1,
            key="model_temperature"
        )
        
        max_tokens = st.slider(
            "Max Tokens:",
            min_value=1024,
            max_value=8192,
            value=8192,
            step=1024,
            key="max_tokens"
        )
        
        if st.button("Apply Settings"):
            st.session_state.agent.llm_config = {
                "temperature": temperature,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": max_tokens,
            }
            st.success("Settings updated!")
    
    # Session management
    with st.expander("Session Management", expanded=False):
        session_filename = st.text_input("Session filename:", "agent_session.json")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Session"):
                st.session_state.agent.save_session(session_filename)
                st.success(f"Session saved to {session_filename}")
        
        with col2:
            if st.button("Load Session"):
                if os.path.exists(session_filename):
                    st.session_state.agent.load_session(session_filename)
                    st.success(f"Session loaded from {session_filename}")
                else:
                    st.error(f"File {session_filename} not found")
    
    # Reset agent
    if st.button("Reset Agent", type="primary"):
        st.session_state.agent.reset_agent()
        st.success("Agent reset successfully!")
        st.experimental_rerun()

# Main content area
st.title("SmartAgent - Task Decomposition")

# Task input section (only shown if no root node exists)
if "root_node_id" not in st.session_state or not st.session_state.root_node_id:
    st.header("Define Your Task")
    
    task_description = st.text_area(
        "Task Description:", 
        height=100,
        placeholder="Describe the complex task you want the agent to decompose..."
    )
    
    constraints = st.text_area(
        "Constraints (one per line):",
        height=80,
        placeholder="format:json\ncontains:subtasks\ncode:python"
    )
    
    if st.button("Start Task", type="primary"):
        if task_description:
            # Parse constraints
            constraint_list = [c for c in constraints.split('\n') if c.strip()]
            # Run the agent
            st.session_state.agent.run(task_description, constraint_list)
            st.success("Task started! Scroll down to see results.")
            st.experimental_rerun()
        else:
            st.error("Please enter a task description before starting.")

# Task tree visualization section (only shown if root node exists)
if "root_node_id" in st.session_state and st.session_state.root_node_id:
    # Show graph visualization
    st.header("Task Hierarchy")
    
    # Render graph using GraphView component
    st.session_state.graph_view.render_graph(
        st.session_state.node_lookup, 
        st.session_state.root_node_id,
        st.session_state.selected_node_id
    )
    
    # Display selected node details
    st.header("Node Details")
    
    # If no node is selected, allow selecting the root node
    if not st.session_state.selected_node_id:
        if st.button("Select Root Node"):
            st.session_state.selected_node_id = st.session_state.root_node_id
            st.experimental_rerun()
    else:
        # Display selected node details and actions
        node = st.session_state.node_lookup[st.session_state.selected_node_id]
        
        # Node information
        st.subheader(f"Selected Node: {node.retrieve_from_memory('task')}")
        
        status_emoji = {
            STATUS_PENDING: "🔘",
            STATUS_RUNNING: "⏳",
            STATUS_COMPLETED: "✅",
            STATUS_FAILED: "❌"
        }.get(node.status, "⚪")
        
        st.write(f"**Status:** {status_emoji} {node.status.capitalize()}")
        
        # Show constraints
        constraints = st.session_state.attention_mechanism.get_constraints(node.node_id)
        if constraints:
            st.write("**Constraints:**")
            for constraint in constraints:
                st.write(f"- {constraint}")
        
        # Show output if available
        if node.output:
            with st.expander("Node Output", expanded=True):
                st.write(node.output)
        
        # Show error if failed
        if node.status == STATUS_FAILED and node.error_message:
            st.error(f"Error: {node.error_message}")
        
        # Node actions based on status
        st.write("**Actions:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if node.status == STATUS_PENDING:
                if st.button("Execute Node", key=f"execute_{node.node_id}"):
                    st.session_state.agent.agentFlow("execute", node)
                    st.experimental_rerun()
            
            if node.status in [STATUS_COMPLETED, STATUS_FAILED]:
                regeneration_guidance = st.text_area(
                    "Regeneration Guidance:", 
                    key=f"guidance_{node.node_id}",
                    placeholder="Provide guidance for regeneration..."
                )
                if st.button("Regenerate", key=f"regenerate_{node.node_id}"):
                    st.session_state.agent.agentFlow("regenerate", node, regeneration_guidance)
                    st.experimental_rerun()
        
        with col2:
            if st.button("Delete Node", key=f"delete_{node.node_id}"):
                if node.node_id == st.session_state.root_node_id:
                    st.session_state.root_node_id = None
                st.session_state.agent.agentFlow("delete", node)
                st.session_state.selected_node_id = None
                st.experimental_rerun()
        
        with col3:
            if st.button("Back to Graph", key=f"back_{node.node_id}"):
                st.session_state.selected_node_id = None
                st.experimental_rerun()
        
        # If node is completed and has children, show them
        if node.status == STATUS_COMPLETED and node.child_ids:
            st.subheader("Child Nodes:")
            child_nodes = [st.session_state.node_lookup[child_id] for child_id in node.child_ids if child_id in st.session_state.node_lookup]
            
            # Group by status
            pending_nodes = [n for n in child_nodes if n.status == STATUS_PENDING]
            completed_nodes = [n for n in child_nodes if n.status == STATUS_COMPLETED]
            failed_nodes = [n for n in child_nodes if n.status == STATUS_FAILED]
            running_nodes = [n for n in child_nodes if n.status == STATUS_RUNNING]
            
            # Show nodes by status groups
            for status_group, nodes, emoji in [
                ("Pending", pending_nodes, "🔘"),
                ("Running", running_nodes, "⏳"),
                ("Completed", completed_nodes, "✅"),
                ("Failed", failed_nodes, "❌")
            ]:
                if nodes:
                    st.write(f"**{emoji} {status_group} ({len(nodes)}):**")
                    for i, child in enumerate(nodes):
                        task_desc = child.retrieve_from_memory("task")
                        if st.button(f"{task_desc}", key=f"child_{child.node_id}"):
                            st.session_state.selected_node_id = child.node_id
                            st.experimental_rerun()
        
        # Add a new child node option
        if node.status == STATUS_COMPLETED:
            st.subheader("Add New Child Node")
            new_task = st.text_area(
                "New Task Description:",
                key=f"new_task_{node.node_id}",
                placeholder="Describe the new subtask..."
            )
            
            if st.button("Add Child Node", key=f"add_child_{node.node_id}"):
                if new_task.strip():
                    child_node = st.session_state.agent.create_child_node(
                        node, 
                        new_task, 
                        node.depth + 1
                    )
                    st.session_state.selected_node_id = child_node.node_id
                    st.experimental_rerun()
                else:
                    st.error("Please enter a task description for the new child node.")

# Footer
st.markdown("---")
st.markdown("SmartAgent - Hierarchical Task Decomposition with Copilot-like editing capabilities")
