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
    
    # Add CSS for fixed terminal at bottom and sticky headers
    st.markdown("""
    <style>
    /* Main container structure */
    .main {
        padding-bottom: 250px !important; /* Make space for the terminal */
    }
    
    
    
    /* Terminal resize handle */
    .terminal-handle {
        position: absolute;
        top: -10px;
        left: 0;
        right: 0;
        height: 1px;
        background-color: #f0f0f0;
        cursor: ns-resize;
        border-top: 1px solid #ddd;
        border-bottom: 1px solid #ddd;
        text-align: center;
    }
    
    /* Terminal handle icon */
    .terminal-handle::after {
        content: "≡";
        font-size: 14px;
        color: #888;
        line-height: 10px;
    }
    
    /* Sticky headers */
    .sticky-header {
        position: sticky;
        top: 0;
        background-color: white;
        z-index: 100;
        padding: 10px 0;
    }
    
    /* Custom styling for sections */
    .editor-section, .agent-section {
        padding: 0 1rem;
        margin-bottom: 0;
    }

    /* Terminal collapse button */
    .terminal-collapse-btn {
        float: right;
        cursor: pointer;
        color: #888;
        font-size: 18px;
        margin-top: -5px;
    }
    
    /* Hide default Streamlit elements that disrupt layout */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Make content areas scrollable */
    .scrollable-content {
        overflow-y: auto;
        max-height: calc(100vh - 350px); /* Adjust based on terminal height */
    }
    </style>
    
    <script>
    // JavaScript for terminal resize functionality
    document.addEventListener('DOMContentLoaded', function() {
        // This script will be loaded but won't work directly in Streamlit
        // We'll need custom components for true drag functionality
        console.log("Terminal resize script loaded");
    });
    </script>
    """, unsafe_allow_html=True)
    
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

    # Main layout - Only top section now
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    # Main content area
    main_container = st.container()
    
    with main_container:
        # Use columns for the main layout
        agent_col, editor_col = st.columns([1, 2])

        # Agent Interface (left column)
        with agent_col:
            # Sticky header for Agent section
            st.markdown('<div class="sticky-header agent-section">', unsafe_allow_html=True)
            st.header("🤖 Smart Agent")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Scrollable content area for agent
            st.markdown('<div class="scrollable-content">', unsafe_allow_html=True)
            
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
                
                # Add toggle for graph view
                if 'show_graph' in st.session_state and st.session_state['show_graph']:
                    from components.ui import render_node_graph
                    root_to_visualize = st.session_state.get('graph_root_id', st.session_state.root_node_id)
                    render_node_graph(root_to_visualize)
                else:
                    from components.ui import render_node_tree
                    render_node_tree(st.session_state.root_node_id)
                
                # Show graph view button at top level
                if not st.session_state.get('show_graph', False) and st.button("Show Full Graph"):
                    st.session_state['show_graph'] = True
                    st.session_state['graph_root_id'] = st.session_state.root_node_id
                    st.experimental_rerun()
                
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
            
            # Close the scrollable content
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Code Editor (middle column)
        with editor_col:
            # Sticky header for Editor section
            st.markdown('<div class="sticky-header editor-section">', unsafe_allow_html=True)
            st.header("💻 Code Editor")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Scrollable content area for editor
            st.markdown('<div class="scrollable-content">', unsafe_allow_html=True)
            
            editor = Editor()
            editor.display()
            
            # Close the scrollable content
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Close the main content
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Terminal (fixed at bottom)
    # Check if terminal should be shown
    if 'show_terminal' not in st.session_state:
        st.session_state.show_terminal = True
    
    if 'terminal_height' not in st.session_state:
        st.session_state.terminal_height = 300
        
    terminal_style = f"""
    <style>
    .fixed-terminal {{
        max-height: {st.session_state.terminal_height}px;
    }}
    </style>
    """
    st.markdown(terminal_style, unsafe_allow_html=True)
    
    if st.session_state.show_terminal:
        st.markdown('<div class="fixed-terminal">', unsafe_allow_html=True)
        st.markdown('<div class="terminal-handle" id="terminal-resize-handle"></div>', unsafe_allow_html=True)
        
        # Terminal header with collapse button
        col1, col2 = st.columns([5, 1])
        with col1:
            st.header("📟 Terminal")
        with col2:
            if st.button("▼", help="Collapse Terminal"):
                st.session_state.show_terminal = False
                st.experimental_rerun()
        
        # Initialize terminal if not already done
        if 'terminal' not in st.session_state:
            st.session_state.terminal = Terminal()
        
        # Display terminal interface
        cli = ClineInterface(st.session_state.terminal)
        cli.display()
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Show the collapsed terminal bar
        st.markdown('''
        <div style="position: fixed; bottom: 0; left: 0; right: 0; height: 30px; 
                    background-color: #f0f0f0; border-top: 1px solid #ddd; 
                    text-align: center; cursor: pointer; z-index: 1000;"
             onclick="document.querySelector('.fixed-terminal').style.display = 'block'; this.style.display = 'none';">
            <span style="line-height: 30px;">📟 Terminal</span>
        </div>
        ''', unsafe_allow_html=True)
        if st.button("▲ Show Terminal", key="show_terminal_btn", help="Expand Terminal"):
            st.session_state.show_terminal = True
            st.experimental_rerun()
    
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
            
        # Terminal settings
        st.subheader("Terminal Settings")
        terminal_height = st.slider("Terminal Height", 100, 600, st.session_state.terminal_height, 50)
        if terminal_height != st.session_state.terminal_height:
            st.session_state.terminal_height = terminal_height
            st.experimental_rerun()
            
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
st.markdown("""
<div style="text-align: center; color: #888; padding-bottom: 350px;">
    <p>AI-Powered IDE with Hierarchical Task Decomposition</p>
    <p>Built with Streamlit and Google Gemini</p>
</div>
""", unsafe_allow_html=True)
