# SMARTAGENT/ui/app.py
import sys
sys.path.insert(0, ".") # Add current directory to Python path
import streamlit as st
import google.generativeai as genai
from agent.agent import Agent
from agent import agent
# from agent.utils import generate_plotly_graph, display_node_textual, display_selected_node_details # Remove Plotly and textual display
from agent.utils import handle_retryable_error, parse_constraint  # Keep utility functions
import os
from dotenv import load_dotenv
from streamlit_agraph import agraph, Node, Edge, Config  # Import agraph components
from agent.memory import LocalMemory


load_dotenv()

st.set_page_config(layout="wide")
st.title("Hierarchical Task Decomposition Agent (Human-Guided)")

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# --- Helper Functions ---
def add_constraint(constraint_type, constraint_value):
    if "initial_constraints" not in st.session_state:
        st.session_state.initial_constraints = []
    st.session_state.initial_constraints.append(f"{constraint_type}: {constraint_value}")

def clear_constraints():
    if "initial_constraints" in st.session_state:
        st.session_state.initial_constraints = []


# --- Main App Logic ---

if 'agent' not in st.session_state:
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
    st.session_state.initial_constraints = []
    st.session_state.task_started = False
    st.session_state.show_regeneration_input = False # Initialize show_regeneration_input


# --- Prompt Library ---
prompt_library = {
    "Create an ebook (PDF) that teaches me trading.": "Generate an ebook PDF that provides a comprehensive guide to trading, covering topics like technical analysis, fundamental analysis, risk management, and trading psychology. Include examples and illustrations.",
    "Write a blog post about sustainable energy.": "Write a blog post about the latest advancements in sustainable energy technologies, focusing on solar, wind, and geothermal power. Discuss the benefits and challenges of each.",
    "Plan a trip to Japan for two weeks.": "Plan a two-week trip to Japan, including an itinerary, suggested accommodations, transportation options, and estimated costs. Consider visiting Tokyo, Kyoto, and Osaka.",
    "Create a Python function to scrape data from a website.": "Create a Python function using libraries like Beautiful Soup and Requests that can scrape data (e.g., product names and prices) from a given website URL.",
    "Generate a marketing plan for a new mobile app.": "Develop a comprehensive marketing plan for a new mobile app, including target audience identification, marketing channels, budget allocation, and key performance indicators (KPIs).",
    "Design a database schema for an e-commerce store.": "Design a database schema for an e-commerce store, including tables for products, customers, orders, and payments. Specify data types and relationships.",
}

# --- Main Chat Input Area ---
task_description = st.chat_input("Enter the initial task:")

# Arrange chat input and start button in a row.
col_input, col_start = st.columns([4, 1])  # Adjust column widths as needed

with col_input:
    selected_prompt = st.selectbox("Choose an example task:", list(prompt_library.keys()))

with col_start:
    st.write("")  # Add some vertical space
    if st.button("Start", type="primary"):
        task_description = prompt_library[selected_prompt]  # Use selected prompt
        if not st.session_state.task_started:
            st.session_state.agent.reset_agent()
            st.session_state.agent.run(task_description)
            st.session_state.selected_node_id = st.session_state.root_node_id
            st.session_state.task_started = True
            st.rerun()

if task_description and not st.session_state.task_started: # This part will run when user enters prompt by pressing Enter.
    st.session_state.agent.reset_agent()
    st.session_state.agent.run(task_description)
    st.session_state.selected_node_id = st.session_state.root_node_id
    st.session_state.task_started = True
    st.rerun()


# --- Sidebar --- (Rest of your sidebar code remains the same) ...
with st.sidebar:
    st.write("## Agent Controls")
    if st.button("Reset Agent"):
        st.session_state.agent.reset_agent()
        st.session_state.task_started = False
        st.rerun()
    if st.session_state.previous_state and st.button("Undo"):
        st.session_state.node_lookup = st.session_state.previous_state["node_lookup"]
        st.session_state.attention_mechanism = st.session_state.previous_state["attention_mechanism"]
        st.session_state.agent.global_memory.update_context(
            st.session_state.previous_state["global_memory"])
        st.session_state.agent.execution_count = st.session_state.previous_state["execution_count"]

        # *** IMPORTANT: Ensure local_memory is correct after undo ***
        for node_id, node in st.session_state.node_lookup.items():
            if not isinstance(node.local_memory, LocalMemory):
                node.local_memory = LocalMemory(node.node_id)

        st.session_state.previous_state = None
        st.session_state.selected_node_id = None
        st.rerun()

    st.write("## Global Context")
    if st.session_state.task_started:
        global_context_summary = st.session_state.agent.global_memory.get_context()
        global_context_summary = global_context_summary[:200] + "..." if len(
            global_context_summary) > 200 else global_context_summary
        st.write(f"{global_context_summary}")

    st.write(f"## Steps: {st.session_state.agent.execution_count}")

    if st.button("View Global Memory"):
        with st.expander("Global Memory"):
            st.write(st.session_state.agent.global_memory.get_context())

    st.write("## Session")
    if st.button("Save"):
        st.session_state.agent.save_session("session.json")
        st.success("Session saved!")

    if st.button("Load"):
        try:
            st.session_state.agent.load_session("session.json")
            st.success("Session loaded!")
            st.session_state.selected_node_id = st.session_state.root_node_id
            st.session_state.task_started = True
        except Exception as e:
            st.error(f"Error loading session: {e}")
        st.rerun()



# --- Main Display Area ---
if st.session_state.task_started:
    col1, col2 = st.columns([2, 1])  # Two-column layout

    with col1:  # Left Column: Node Details
        st.write("## Task Hierarchy")
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
                        color = "#f0b8a8" # Light purple
                    else:
                        color = "#f0d8a8" # Light orange


                    # --- Node Styling ---
                    node_style = {
                        "size": 25,
                        "shape": "circularImage",
                        "color": color,
                        "image": "https://cdn-icons-png.flaticon.com/512/8859/8859891.png",  # Your icon
                        "label": f"{node.node_id[:8]}...",
                        "title": node.task_description,  # Tooltip: Task Description
                    }

                    # Highlight selected node
                    if node_id == st.session_state.selected_node_id:
                        node_style["color"] = "#ff0000"  # Example: Red for selected
                        node_style["size"] = 35  # Larger size for selected
                        # You could add a border here if needed, but it often looks cluttered

                    nodes.append(Node(id=node_id, **node_style))
                    # --- End Node Styling ---

                    if node.parent_id:
                        edges.append(Edge(source=node.parent_id, target=node_id, type="CURVE_SMOOTH"))

                config = Config(width=750,
                                height=500,
                                directed=True,
                                physics=False,  # Disable physics for hierarchical layout
                                hierarchical=True, # THIS IS KEY for a tree layout
                                # --- Hover Styling ---
                                nodeHighlightBehavior=True,  # Enable node highlighting
                                highlightColor="#f0f0f0",  # Light gray on hover
                                # --- End Hover Styling ---
                                )

                return_value = agraph(nodes=nodes, edges=edges, config=config)

                if return_value:
                    st.session_state.selected_node_id = return_value
                    st.rerun()

            except Exception as e:
                st.error(f"Error generating graph: {e}")
        else:
            st.write("No nodes to display yet.")


    with col2:  # Right Column: Node Details
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
                    for constraint in st.session_state.attention_mechanism.get_constraints(node.node_id):
                        st.write(f"- {constraint}")

                with st.expander("Dependencies"):
                    for dep_id in st.session_state.attention_mechanism.get_dependencies(node.node_id):
                        dep_node = st.session_state.node_lookup.get(dep_id)
                        if dep_node:
                            st.write(f"- Depends on: {dep_node.node_id} ({dep_node.task_description})")
                        else:
                            st.write(f"- Depends on: {dep_id} (NOT FOUND)")

                st.write(f"**Output:**")
                st.write(node.output)
                if node.error_message:
                    st.error(f"**Error:** {node.error_message}")

                if node.status == "pending":
                    if st.button("Execute Node", key=f"execute_{node.node_id}", type="primary"):
                        st.session_state.previous_state = {
                            "node_lookup": st.session_state.node_lookup.copy(),
                            "attention_mechanism": st.session_state.attention_mechanism,
                            "global_memory": st.session_state.agent.global_memory.get_context(),
                            "execution_count": st.session_state.agent.execution_count
                        }
                        st.session_state.agent.agentFlow("execute", node)
                        st.rerun()

                if node.status in ("running", "completed", "failed", "overridden"):
                    if st.button("Regenerate", key=f"regenerate_{node.node_id}"):
                        st.session_state.show_regeneration_input = True

                    if st.session_state.get("show_regeneration_input"):
                        regeneration_guidance = st.text_area("Enter regeneration guidance (optional):", key=f"guidance_{node_id}")
                        if st.button("Submit Guidance", key=f"submit_guidance_{node_id}"):
                            st.session_state.previous_state = {
                                "node_lookup": st.session_state.node_lookup.copy(),
                                "attention_mechanism": st.session_state.attention_mechanism,
                                "global_memory": st.session_state.agent.global_memory.get_context(),
                                "execution_count": st.session_state.agent.execution_count
                            }
                            st.session_state.agent.agentFlow("regenerate", node, regeneration_guidance)
                            st.session_state.show_regeneration_input = False
                            st.rerun()

                if st.button("Delete Node", key=f"delete_{node.node_id}"):
                    if st.session_state.get(f"confirm_delete_{node_id}", False):
                        st.session_state.previous_state = {
                            "node_lookup": st.session_state.node_lookup.copy(),
                            "attention_mechanism": st.session_state.attention_mechanism,
                            "global_memory": st.session_state.agent.global_memory.get_context(),
                            "execution_count": st.session_state.agent.execution_count
                        }
                        st.session_state.agent.agentFlow("delete", node)
                        st.session_state.selected_node_id = None
                        st.session_state[f"confirm_delete_{node_id}"] = False
                        st.rerun()
                    else:
                        st.warning(f"Are you sure you want to delete Node {node_id} and all its children?")
                        if st.button("Confirm Delete", key=f"confirm_{node_id}"):
                            st.session_state[f"confirm_delete_{node_id}"] = True
                            st.rerun()
            else:
                st.info("Select a node") # Added this line to handle cases where node is not found

        else:
            st.info("Select a node in the graph to view details.")
else:
    st.write("Enter a task in the chat input to start.")
