import sys
sys.path.insert(0, ".")  # Add current directory to Python path
import streamlit as st
import os
import uuid
import json
from typing import Optional, Dict, List
from agent.node import Node
from agent.memory import LocalMemory, GlobalMemory
from agent.llm_client import LLMClient
from dotenv import load_dotenv
from streamlit_agraph import agraph, Node as ANode, Edge, Config

# Load environment variables from .env file
load_dotenv()

# Configure page
st.set_page_config(
    page_title="SmartAgent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- AttentionMechanism and Agent Class Definitions (use your existing classes) ---
from agent.attention_mechanism import AttentionMechanism

class Agent:
    def __init__(self, max_depth: int = 3):
        self.root_node_id: Optional[str] = None
        self.max_depth: int = max_depth
        self.global_memory = GlobalMemory()
        self.execution_count = 0
        # Summary interval for global context updating
        self.global_context_summary_interval = 5
    
    def create_root_node(self, task_description: str) -> Node:
        node = Node(None, task_description, 0)
        st.session_state.node_lookup[node.node_id] = node
        self.root_node_id = node.node_id
        return node
    
    def create_child_node(self, parent_node: Node, task_description: str, depth: int) -> Node:
        child = Node(parent_node.node_id, task_description, depth)
        st.session_state.node_lookup[child.node_id] = child
        parent_node.add_child(child)
        return child
    
    def delete_node_and_children(self, node: Node) -> None:
        """Recursively deletes a node and all its descendants."""
        # Remove children first
        for child_id in node.child_ids:
            if child_id in st.session_state.node_lookup:
                self.delete_node_and_children(st.session_state.node_lookup[child_id])

        # Now remove the node itself
        st.session_state.node_lookup.pop(node.node_id, None)
        st.session_state.attention_mechanism.remove_node(node.node_id)
    
    def agentFlow(self, action: str, node: Node, regeneration_guidance: str = "") -> None:
        """
        Centralized function to handle node actions (execution, regeneration, deletion).
        """
        if action == "execute":
            process_node(node)
        elif action == "regenerate":
            self._regenerate_node(node, regeneration_guidance)
        elif action == "delete":
            self.delete_node_and_children(node)
        else:
            raise ValueError(f"Invalid action: {action}")
    
    def _regenerate_node(self, node: Node, regeneration_guidance: str) -> None:
        """Resets a node for regeneration and stores guidance."""
        node.status = "pending"
        node.output = ""
        node.error_message = ""
        # Ensure local_memory is a LocalMemory object
        if not isinstance(node.local_memory, LocalMemory):
            node.local_memory = LocalMemory(node.node_id)  # Re-initialize if needed
        node.store_in_memory("regeneration_guidance", regeneration_guidance)
    
    def reset_agent(self) -> None:
        """Resets the agent's state."""
        st.session_state.node_lookup = {}
        self.root_node_id = None
        self.execution_count = 0
        self.global_memory = GlobalMemory()
        st.session_state.selected_node_id = None  # Reset selected node
        st.session_state.previous_state = None  # Reset Undo

    def save_session(self, filename: str) -> None:
        """Saves the current session to a JSON file."""
        data = {
            "node_lookup": {node_id: node.__dict__ for node_id, node in st.session_state.node_lookup.items()},
            "attention_mechanism": {
                "dependency_graph": st.session_state.attention_mechanism.dependency_graph 
                    if hasattr(st.session_state.attention_mechanism, 'dependency_graph') else {},
                "constraints": st.session_state.attention_mechanism.constraints,
            },
            "root_node_id": st.session_state.get('root_node_id'),
            "global_memory": self.global_memory.global_context,  # Save Global Memory
            "execution_count": self.execution_count  # Save Execution Count
        }

        # Convert Node objects to dictionaries (and LocalMemory)
        for node_id, node_data in data["node_lookup"].items():
            node_data["local_memory"] = node_data["local_memory"].local_memory

        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

    def load_session(self, filename: str) -> None:
        """Loads a session from a JSON file."""
        with open(filename, "r") as f:
            data = json.load(f)

        # Clear existing state
        self.reset_agent()

        st.session_state.root_node_id = data["root_node_id"]
        # Re-create Node objects and LocalMemory
        st.session_state.node_lookup = {}
        for node_id, node_data in data["node_lookup"].items():
            new_node = Node(node_data["parent_id"], node_data["task_description"], node_data["depth"])
            new_node.node_id = node_data["node_id"]
            new_node.child_ids = node_data["child_ids"]
            new_node.status = node_data["status"]
            new_node.output = node_data["output"]
            new_node.error_message = node_data["error_message"]
            # *** IMPORTANT: Create a LocalMemory object ***
            new_node.local_memory = LocalMemory(new_node.node_id)  # Create the object
            new_node.local_memory.local_memory = node_data["local_memory"]  # Load the data
            st.session_state.node_lookup[node_id] = new_node

        st.session_state.attention_mechanism.dependency_graph = data["attention_mechanism"]["dependency_graph"]
        st.session_state.attention_mechanism.constraints = data["attention_mechanism"]["constraints"]
        self.global_memory.update_context(data["global_memory"])  # Load Global Memory
        self.execution_count = data["execution_count"]  # Load Execution Count

# --- Helper Functions ---

def initialize_session():
    """Initialize session state variables"""
    if "node_lookup" not in st.session_state:
        st.session_state.node_lookup = {}
    
    if "attention_mechanism" not in st.session_state:
        st.session_state.attention_mechanism = AttentionMechanism()
    
    if "agent" not in st.session_state:
        st.session_state.agent = Agent()
        # Set default global context
        st.session_state.agent.global_memory.update_context("This agent decomposes complex tasks.")
    
    if "llm_client" not in st.session_state:
        st.session_state.llm_client = LLMClient(provider="google")  # Default to Google provider
    
    if "selected_node_id" not in st.session_state:
        st.session_state.selected_node_id = None
        
    if "task_started" not in st.session_state:
        st.session_state.task_started = False
    
    if "show_regeneration_input" not in st.session_state:
        st.session_state.show_regeneration_input = False
    
    # Initialize state for raw response toggles
    for node_id in st.session_state.node_lookup:
        if f"show_raw_{node_id}" not in st.session_state:
            st.session_state[f"show_raw_{node_id}"] = False


def process_node(node: Node):
    """Process a node with the LLM and enhanced error handling"""
    node.status = "running"
    
    # Get the prompt
    original_prompt = node.build_prompt()
    
    # Add even clearer formatting guidance
    formatting_reminder = """
CRITICAL INSTRUCTION: Your response MUST strictly follow one of these two JSON formats and no other:

1. For direct solutions - ALWAYS USE THIS FORMAT WHEN YOU HAVE A SOLUTION:
```json
{
  "result": "Your detailed solution here with all relevant information"
}
```

2. For task decomposition - ONLY USE THIS FORMAT WHEN BREAKING DOWN INTO SUBTASKS:
```json
{
  "subtasks": [
    {
      "task_description": "Detailed description of subtask 1",
      "constraints": []
    },
    {
      "task_description": "Detailed description of subtask 2",
      "constraints": []
    }
  ]
}
```

IMPORTANT:
- DO NOT use other top-level JSON keys like "task_description" at the root level
- DO NOT include explanations outside the JSON
- Put ALL your analysis inside the "result" field if you're providing a direct solution
- Make sure the JSON is properly formatted with double quotes around all keys
"""
    
    prompt = original_prompt + "\n" + formatting_reminder
    
    # Show a loading spinner
    with st.spinner(f"Processing node: {node.task_description}"):
        try:
            # Call the LLM
            llm_response = st.session_state.llm_client.complete(prompt)
            
            # Store the raw response in node memory
            node.store_in_memory("raw_llm_output", llm_response)
            
            # Process the response
            node.process_llm_output(llm_response)
            
            # Update status
            if node.error_message:
                node.status = "failed"
            else:
                node.status = "completed"
            
            # Update execution count and possibly summarize
            st.session_state.agent.execution_count += 1
        except Exception as e:
            node.error_message = f"Error processing node: {str(e)}"
            node.status = "failed"
            st.error(f"Exception during node processing: {str(e)}")
    
    return node


# --- Prompt Library ---
PROMPT_LIBRARY = {
    "Create a To-Do List App": "Create a simple command-line to-do list app where users can add tasks, view all tasks, mark tasks as completed, delete tasks, and save tasks to a file.",
    "Analyze Stock Market Data": "Develop a Python script that analyzes stock market data using pandas and matplotlib, including visualization of trends and basic predictive analysis.",
    "Build a Weather App": "Create a weather application that fetches data from a weather API and displays current conditions and forecasts for any city.",
    "Design a Database Schema": "Design a database schema for an e-commerce platform with users, products, orders, and payment information.",
    "Create a Machine Learning Model": "Develop a machine learning model to classify text as positive or negative sentiment using scikit-learn and NLTK.",
    "Build a Portfolio Website": "Create a responsive personal portfolio website using HTML, CSS, and JavaScript with sections for about, projects, and contact information."
}


# --- UI Components ---

def render_node_graph():
    """Render the node hierarchy as an interactive graph"""
    if st.session_state.node_lookup:
        try:
            nodes = []
            edges = []
            for node_id, node in st.session_state.node_lookup.items():
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

                # Node styling
                node_style = {
                    "size": 25,
                    "shape": "circularImage",
                    "color": color,
                    "image": "https://cdn-icons-png.flaticon.com/512/8859/8859891.png",  # Task icon
                    "label": f"{node_id[:8]}...",
                    "title": node.task_description,  # Tooltip: Task Description
                }

                # Highlight selected node
                if node_id == st.session_state.selected_node_id:
                    node_style["color"] = "#ff0000"  # Red for selected
                    node_style["size"] = 35  # Larger size for selected

                nodes.append(ANode(id=node_id, **node_style))

                if node.parent_id:
                    edges.append(Edge(source=node.parent_id, target=node_id, type="CURVE_SMOOTH"))

            config = Config(width=750,
                            height=500,
                            directed=True,
                            physics=False,  # Disable physics for hierarchical layout
                            hierarchical=True,  # Tree layout
                            nodeHighlightBehavior=True,
                            highlightColor="#f0f0f0",
                            )

            return_value = agraph(nodes=nodes, edges=edges, config=config)

            if return_value:
                st.session_state.selected_node_id = return_value
                st.rerun()

        except Exception as e:
            st.error(f"Error generating graph: {e}")
    else:
        st.write("No nodes to display yet.")


def render_node_details():
    """Render detailed information about the selected node"""
    if st.session_state.selected_node_id:
        node = st.session_state.node_lookup.get(st.session_state.selected_node_id)
        if node:
            # Node Status Display
            if node.status == "pending":
                st.info(f"Ready to Execute", icon="✅")
            elif node.status == "running":
                st.info(f"Executing...", icon="⏳")
            elif node.status == "completed":
                st.success(f"Completed", icon="✅")
            elif node.status == "failed":
                st.error(f"Failed", icon="❌")
            elif node.status == "overridden":
                st.warning(f"Overridden", icon="⚠️")

            # Node Details Display
            st.write(f"**Task:** {node.task_description}")

            with st.expander("Constraints"):
                constraints = st.session_state.attention_mechanism.get_constraints(node.node_id)
                if constraints:
                    for constraint in constraints:
                        st.write(f"- {constraint}")
                else:
                    st.write("No constraints defined.")

            with st.expander("Dependencies"):
                dependencies = st.session_state.attention_mechanism.get_dependencies(node.node_id)
                if dependencies:
                    for dep_id in dependencies:
                        if dep_id:  # Check if dependency is not None
                            dep_node = st.session_state.node_lookup.get(dep_id)
                            if dep_node:
                                st.write(f"- Depends on: {dep_id[:8]}... ({dep_node.task_description})")
                            else:
                                st.write(f"- Depends on: {dep_id[:8]}... (NOT FOUND)")
                else:
                    st.write("No dependencies.")

            st.write(f"**Output:**")
            if node.output:
                st.write(node.output)
            else:
                st.write("No output yet.")
                
            if node.error_message:
                st.error(f"**Error:** {node.error_message}")

            # Raw Response Viewer
            raw_response = node.retrieve_from_memory("raw_llm_response")
            if raw_response:
                with st.expander("Raw LLM Response"):
                    st.code(raw_response, language="text")

            # Node Actions
            st.write("**Actions:**")
            action_cols = st.columns(3)
            
            # Execute button (pending nodes only)
            if node.status == "pending":
                if action_cols[0].button("Execute Node", key=f"execute_{node.node_id}", type="primary"):
                    st.session_state.previous_state = {
                        "node_lookup": st.session_state.node_lookup.copy(),
                        "attention_mechanism": st.session_state.attention_mechanism,
                        "global_memory": st.session_state.agent.global_memory.get_context(),
                        "execution_count": st.session_state.agent.execution_count
                    }
                    st.session_state.agent.agentFlow("execute", node)
                    st.rerun()

            # Regenerate button
            if node.status in ("completed", "failed", "overridden"):
                if action_cols[1].button("Regenerate", key=f"regenerate_{node.node_id}"):
                    st.session_state.show_regeneration_input = True

                if st.session_state.show_regeneration_input:
                    regeneration_guidance = st.text_area("Enter regeneration guidance (optional):", key=f"guidance_{node.node_id}")
                    if st.button("Submit Guidance", key=f"submit_guidance_{node.node_id}"):
                        st.session_state.previous_state = {
                            "node_lookup": st.session_state.node_lookup.copy(),
                            "attention_mechanism": st.session_state.attention_mechanism,
                            "global_memory": st.session_state.agent.global_memory.get_context(),
                            "execution_count": st.session_state.agent.execution_count
                        }
                        st.session_state.agent.agentFlow("regenerate", node, regeneration_guidance)
                        st.session_state.show_regeneration_input = False
                        st.rerun()

            # View Prompt button
            if action_cols[2].button("View Prompt", key=f"prompt_{node.node_id}"):
                with st.expander("Prompt", expanded=True):
                    st.code(node.build_prompt(), language="markdown")

            # Delete button
            if st.button("Delete Node", key=f"delete_{node.node_id}", type="secondary"):
                delete_key = f"confirm_delete_{node.node_id}"
                if st.session_state.get(delete_key, False):
                    st.session_state.previous_state = {
                        "node_lookup": st.session_state.node_lookup.copy(),
                        "attention_mechanism": st.session_state.attention_mechanism,
                        "global_memory": st.session_state.agent.global_memory.get_context(),
                        "execution_count": st.session_state.agent.execution_count
                    }
                    st.session_state.agent.agentFlow("delete", node)
                    st.session_state.selected_node_id = None
                    st.session_state[delete_key] = False
                    st.rerun()
                else:
                    st.warning(f"Are you sure you want to delete this node and all its children?")
                    if st.button("Confirm Delete", key=f"confirm_{node.node_id}"):
                        st.session_state[delete_key] = True
                        st.rerun()

        else:
            st.info("Selected node not found. It may have been deleted.")
    else:
        st.info("Select a node in the graph to view details.")


# --- Main App Logic ---

def main():
    st.title("🤖 SmartAgent - Visual Task Decomposition")
    
    initialize_session()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Provider selection
        provider = st.selectbox(
            "LLM Provider",
            options=["google", "openai"],
            index=0,  # Default to Google
            key="provider_select"
        )
        
        if provider != st.session_state.llm_client.provider:
            st.session_state.llm_client.set_provider(provider)
            st.rerun()  # Refresh to update available models
        
        # API Key input
        api_key_label = "Google API Key" if provider == "google" else "OpenAI API Key"
        api_key = st.text_input(
            api_key_label,
            type="password",
            help=f"Enter your {provider.capitalize()} API key"
        )
        if api_key:
            st.session_state.llm_client.set_api_key(api_key)
        
        # Model selection - dynamically update based on provider
        available_models = st.session_state.llm_client.get_available_models()
        default_index = 0
        
        model = st.selectbox(
            "LLM Model",
            options=available_models,
            index=default_index
        )
        st.session_state.llm_client.set_model(model)
        
        # If using Google, show additional information about Gemini models
        if provider == "google":
            with st.expander("Gemini Model Information"):
                st.markdown("""
                **Gemini Model Capabilities:**
                - **gemini-2.0-flash-thinking-exp-01-21**: Optimized for complex reasoning
                - **gemini-2.0-pro-exp-02-05**: Best for comprehensive tasks
                - **gemini-2.0-flash-lite**: Fastest, best for simple tasks
                - **gemini-2.0-flash**: Good balance of speed and quality
                - **gemini-2.0-flash-exp**: Experimental version with image generation capabilities
                """)
        
        # Global context input - Modified to use agent.global_memory instead of attention_mechanism
        global_context = st.text_area(
            "Global Context",
            value=st.session_state.agent.global_memory.get_context(),
            height=150
        )
        # Update global context in the agent's global memory instead
        st.session_state.agent.global_memory.update_context(global_context)
        
        # Max depth setting
        max_depth = st.slider(
            "Maximum Task Depth",
            min_value=1,
            max_value=5,
            value=st.session_state.agent.max_depth
        )
        st.session_state.agent.max_depth = max_depth
        
        # Session management
        st.write("## Session")
        session_cols = st.columns(2)
        
        if session_cols[0].button("Save Session"):
            st.session_state.agent.save_session("session.json")
            st.success("Session saved!")

        if session_cols[1].button("Load Session"):
            try:
                if os.path.exists("session.json"):
                    st.session_state.agent.load_session("session.json")
                    st.session_state.task_started = True
                    st.success("Session loaded!")
                else:
                    st.warning("No saved session found.")
            except Exception as e:
                st.error(f"Error loading session: {e}")
        
        # Reset button
        if st.button("Reset Agent", type="secondary"):
            st.session_state.agent.reset_agent()
            st.session_state.task_started = False
            st.rerun()
        
        # Debug options
        st.write("---")
        with st.expander("Debug Options", expanded=False):
            if st.button("Force JSON Mode"):
                # Add a special instruction to the global context
                current_context = st.session_state.agent.global_memory.get_context()
                json_instruction = "\n\nIMPORTANT: All responses MUST be in valid JSON format with either a 'result' key or 'subtasks' key."
                st.session_state.agent.global_memory.update_context(current_context + json_instruction)
                st.success("Added JSON formatting instruction to global context")
    
    # Main task input area - only show if no task has been started
    if not st.session_state.task_started:
        st.write("## Create a new task")
        
        # Example tasks dropdown
        selected_example = st.selectbox(
            "Choose an example task:",
            options=list(PROMPT_LIBRARY.keys()),
            index=0
        )
        
        # Task description input - either manual or from example
        task_description = st.text_area(
            "Or enter your own task description:", 
            height=100,
            value=PROMPT_LIBRARY[selected_example]
        )
        
        # Create task button
        if st.button("Create Task", type="primary"):
            if task_description:
                root_node = st.session_state.agent.create_root_node(task_description)
                st.session_state.root_node_id = root_node.node_id
                st.session_state.selected_node_id = root_node.node_id
                st.session_state.task_started = True
                st.rerun()
            else:
                st.error("Please enter a task description.")
    else:
        # Show the task execution interface once a task has been started
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.write("## Task Hierarchy")
            render_node_graph()
            
        with col2:
            st.write("## Node Details")
            render_node_details()

if __name__ == "__main__":
    main()
