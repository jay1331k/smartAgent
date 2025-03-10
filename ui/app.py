# SMARTAGENT/ui/app.py (Corrected)
import streamlit as st
import google.generativeai as genai
from agent.agent import Agent
# NEW: Import constants directly from the agent module
from agent import agent
from agent.utils import generate_plotly_graph, display_node_textual, display_selected_node_details
import os
from dotenv import load_dotenv
import plotly.graph_objects as go

load_dotenv()

st.set_page_config(layout="wide")
st.title("Hierarchical Task Decomposition Agent (Human-Guided)")

# Configure Gemini API Key
genai.configure(api_key=os.environ["GEMINI_API_KEY"])


# --- Helper Functions ---
def add_constraint(constraint_type, constraint_value):
    """Adds a constraint to the list of initial constraints."""
    if "initial_constraints" not in st.session_state:
        st.session_state.initial_constraints = []
    st.session_state.initial_constraints.append(f"{constraint_type}: {constraint_value}")


def clear_constraints():
    """Clears the list of initial constraints."""
    if "initial_constraints" in st.session_state:
        st.session_state.initial_constraints = []


# --- Main App Logic ---

if 'agent' not in st.session_state:
    # Use the constants directly from the module
    llm = genai.GenerativeModel(model_name=agent.LLM_MODEL)
    llm_config = genai.GenerationConfig(
        temperature=agent.LLM_TEMPERATURE,
        max_output_tokens=agent.LLM_MAX_TOKENS
    )
    st.session_state.llm_config = llm_config
    st.session_state.agent = Agent(llm=llm, llm_config=llm_config)
    st.session_state.agent.setup_agent()
    st.session_state.selected_node_id = None
    st.session_state.previous_state = None
    st.session_state.initial_constraints = []  # Initialize initial constraints

with st.form("task_input_form"):
    task_description = st.text_input("Enter the initial task:", "MAKE A LUDO GAME THAT I CAN ENJOY WITH MY FRIENDS",
                                     max_chars=250)

    # --- Structured Constraint Input ---
    with st.expander("Add Constraints (Optional)"):
        constraint_type = st.selectbox("Constraint Type", ["format", "contains", "max_length"])
        constraint_value = st.text_input("Constraint Value")
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Add Constraint", key="add_constraint_btn"):
                add_constraint(constraint_type, constraint_value)
        with col2:
            if st.button("Clear Constraints", key="clear_constraints_btn"):
                clear_constraints()

        st.write("Current Constraints:")
        if "initial_constraints" in st.session_state:
            for constraint in st.session_state.initial_constraints:
                st.write(f"- {constraint}")
    # ---

    submitted = st.form_submit_button("Start Agent")

if submitted:
    st.session_state.agent.reset_agent()
    # Use the list of initial constraints directly
    constraints = st.session_state.get("initial_constraints", [])
    st.session_state.agent.run(task_description, constraints)
    st.session_state.selected_node_id = st.session_state.root_node_id
    st.session_state.initial_constraints = []  # Clear Constraints
    st.rerun()

# --- Status Message ---
# Improved status message to clearly indicate "Ready to Execute"
if st.session_state.selected_node_id:
    node = st.session_state.node_lookup.get(st.session_state.selected_node_id)
    if node:
        if node.status == "pending":
            st.info(f"Node {node.node_id} is ready for execution. Click 'Execute Node' to proceed.")
        elif node.status == "running":
            st.info(f"Node {node.node_id} is currently executing.")
        elif node.status == "completed":
            st.success(f"Node {node.node_id} has completed execution.")
        elif node.status == "failed":
            st.error(f"Node {node.node_id} has failed. See details below for error message.")
        elif node.status == "overridden":
            st.warning(f"Node {node.node_id} was overridden.")
    else:  # Handle if node is not found(deleted)
        st.info("Select a node to begin.")
else:
    st.info("Enter a task and click 'Start Agent' to begin.")


# --- Global Context Summary ---
global_context_summary = st.session_state.agent.global_memory.get_context()
global_context_summary = global_context_summary[:200] + "..." if len(
    global_context_summary) > 200 else global_context_summary
st.write(f"**Global Context Summary:** {global_context_summary}")

# --- Step Counter ---
st.write(f"## Step Count: {st.session_state.agent.execution_count}")

# --- Main Display Area ---
col1, col2 = st.columns([2, 1])

with col1:
    st.write("## Node Hierarchy (Interactive Graph):")
    try:
        fig = generate_plotly_graph()
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error generating graph: {e}")

    st.write("## Node Hierarchy (Textual):")
    if st.session_state.get("root_node_id"):
        display_node_textual(st.session_state.root_node_id, st.session_state.selected_node_id)

with col2:
    if st.session_state.selected_node_id:
        display_selected_node_details(st.session_state.selected_node_id)
    else:
        st.info("Select a node in the graph or list to view details.")

# --- Global Context and Agent Control ---

col_control1, col_control2, col_control3 = st.columns(3)

with col_control1:
    if st.button("View Global Memory"):
        with st.expander("Global Memory"):
            st.write(st.session_state.agent.global_memory.get_context())

with col_control2:
    if st.button("Reset Agent"):
        st.session_state.agent.reset_agent()
        st.rerun()

with col_control3:
    if st.session_state.previous_state and st.button("Undo"):
        st.session_state.node_lookup = st.session_state.previous_state["node_lookup"]
        st.session_state.attention_mechanism = st.session_state.previous_state["attention_mechanism"]
        st.session_state.agent.global_memory.update_context(
            st.session_state.previous_state["global_memory"])  # Undo GM
        st.session_state.agent.execution_count = st.session_state.previous_state["execution_count"]  # Undo EC
        st.session_state.previous_state = None
        st.session_state.selected_node_id = None
        st.rerun()

# Session History
col_sh1, col_sh2 = st.columns(2)

with col_sh1:
    if st.button("Save Session"):
        st.session_state.agent.save_session("session.json")
        st.success("Session saved!")

with col_sh2:
    if st.button("Load Session"):
        try:
            st.session_state.agent.load_session("session.json")
            st.success("Session loaded!")
            st.session_state.selected_node_id = st.session_state.root_node_id  # Select root
        except Exception as e:
            st.error(f"Error loading session: {e}")
        st.rerun()