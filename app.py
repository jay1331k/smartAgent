import streamlit as st
import os
import sys
from pathlib import Path

# Add the project directory to the system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import components
from components.agent import Agent
from components.node import Node
from components.file_explorer import FileExplorer
from components.terminal import Terminal, ClineInterface
from components.editor import Editor
from components.utils import initialize_gemini_api, get_model, STATUS_PENDING, STATUS_COMPLETED

# Try to import any useful classes/functions from agent folder if they exist
try:
    from agent.constants import *
    from agent.node import Node as AgentNode
    from agent.memory import LocalMemory as AgentLocalMemory
    from agent.memory import GlobalMemory as AgentGlobalMemory
except ImportError:
    pass  # If these don't exist, we'll use our component versions

def main():
    st.set_page_config(page_title="Smart Agent IDE", layout="wide", page_icon="🤖")
    
    # Initialize Gemini API
    if not initialize_gemini_api():
        st.stop()
    
    # Setup sidebar (file explorer)
    st.sidebar.title("📁 File Explorer")
    
    # Directory input in sidebar
    default_dir = st.session_state.get("explorer_dir", os.getcwd())
    dir_path = st.sidebar.text_input("Directory Path:", value=default_dir)
    if st.sidebar.button("Browse"):
        st.session_state.explorer_dir = dir_path
        st.experimental_rerun()
    
    # File explorer in sidebar
    if os.path.isdir(dir_path):
        explorer = FileExplorer(dir_path)
        with st.sidebar:
            explorer.display_file_tree()
    else:
        st.sidebar.error("Invalid directory path")
    
    # Main layout - divide into top and bottom sections
    main_top, main_bottom = st.container(), st.container()

    # Main top - divide into agent and editor
    with main_top:
        agent_col, editor_col = st.columns([1, 2])
        
        # Agent Interface (left column)
        with agent_col:
            st.header("🤖 Smart Agent")
            
            # Initialize Agent
            if 'agent' not in st.session_state:
                llm = get_model()
                if not llm:
                    st.error("Failed to initialize LLM model")
                    st.stop()
                
                llm_config = {
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 2048,
                }
                
                agent = Agent(
                    llm=llm,
                    llm_config=llm_config,
                    global_context="This agent can break down complex tasks into manageable subtasks."
                )
                st.session_state.agent = agent
                agent.setup_agent()
            
            # Task input and running
            if 'root_node_id' not in st.session_state:
                with st.form("task_form"):
                    task_description = st.text_area("Enter Task Description:", height=100)
                    initial_constraint = st.text_input("Initial Constraint (optional):")
                    submitted = st.form_submit_button("Run Task")
                    
                    if submitted and task_description:
                        initial_constraints = [initial_constraint] if initial_constraint else None
                        st.session_state.agent.run(task_description, initial_constraints)
                        st.experimental_rerun()
            # Display task tree
            else:
                st.subheader("Task Tree")
                from components.ui import render_node_tree
                render_node_tree(st.session_state.root_node_id)
                
                # New task button
                if st.button("Start New Task"):
                    st.session_state.agent.reset_agent()
                    st.experimental_rerun()
                
                # Save/Load session
                col1, col2 = st.columns(2)
                with col1:
                    save_filename = st.text_input("Save session as:", value="agent_session.json")
                    if st.button("Save Session"):
                        st.session_state.agent.save_session(save_filename)
                        st.success(f"Session saved to {save_filename}")
                
                with col2:
                    uploaded_file = st.file_uploader("Load session from file:", type="json")
                    if uploaded_file and st.button("Load Session"):
                        from tempfile import NamedTemporaryFile
                        with NamedTemporaryFile(delete=False, suffix=".json") as tmp:
                            tmp.write(uploaded_file.getvalue())
                            tmp_path = tmp.name
                        
                        try:
                            st.session_state.agent.load_session(tmp_path)
                            st.success("Session loaded successfully")
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Error loading session: {str(e)}")
                        finally:
                            os.unlink(tmp_path)
        
        # Code Editor (middle column)
        with editor_col:
            editor = Editor()
            editor.display()
    
    # Terminal (bottom section)
    with main_bottom:
        st.header("📟 Terminal")
        if 'terminal' not in st.session_state:
            st.session_state.terminal = Terminal()
        
        cli = ClineInterface(st.session_state.terminal)
        cli.display()
    
    # Settings in sidebar
    with st.sidebar.expander("⚙️ Settings"):
        # Model selection
        st.subheader("Model Settings")
        models = [
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-2.0-pro-exp-02-05", 
            "gemini-2.0-flash-thinking-exp-01-21",
            "gemini-2.0-flash-exp"
        ]
        selected_model = st.selectbox("Select Model", models, 
                                       index=models.index("gemini-2.0-pro-exp-02-05") if "gemini-2.0-pro-exp-02-05" in models else 0)
        if selected_model != st.session_state.get('selected_model'):
            st.session_state.selected_model = selected_model

        # LLM configuration
        st.subheader("LLM Configuration")
        temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.get("llm_config", {}).get("temperature", 0.7), 0.1)
        top_p = st.slider("Top P", 0.0, 1.0, st.session_state.get("llm_config", {}).get("top_p", 0.95), 0.05)
        top_k = st.slider("Top K", 1, 100, st.session_state.get("llm_config", {}).get("top_k", 40), 1)
        max_tokens = st.slider("Max Output Tokens", 100, 8192, st.session_state.get("llm_config", {}).get("max_output_tokens", 2048), 100)
        
        if st.button("Apply Settings"):
            st.session_state.llm_config = {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_output_tokens": max_tokens,
            }
            if 'agent' in st.session_state:
                st.session_state.agent.llm = get_model()
                st.session_state.agent.llm_config = st.session_state.llm_config
            st.success("Settings applied successfully")
            
        # Agent configuration
        st.subheader("Agent Configuration")
        if 'agent' in st.session_state:
            max_depth = st.slider("Max Tree Depth", 1, 10, st.session_state.agent.max_depth, 1)
            summary_interval = st.slider("Global Context Summary Interval", 1, 20, st.session_state.agent.global_context_summary_interval, 1)
            
            if st.button("Apply Agent Settings"):
                st.session_state.agent.max_depth = max_depth
                st.session_state.agent.global_context_summary_interval = summary_interval
                st.success("Agent settings applied successfully")

# Run the app
if __name__ == "__main__":
    main()

# Add a footer with app information
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    <p>AI-Powered IDE with Hierarchical Task Decomposition</p>
    <p>Built with Streamlit and Google Gemini</p>
</div>
""", unsafe_allow_html=True)
