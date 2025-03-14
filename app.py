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
from components.graph_view import GraphView
from components.utils import initialize_gemini_api, get_model, STATUS_PENDING, STATUS_COMPLETED, STATUS_RUNNING, STATUS_FAILED

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
    
    /* Node status colors */
    .node-pending {
        background-color: #f8f9fa;
        border-left: 5px solid #6c757d;
    }
    .node-running {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
    }
    .node-completed {
        background-color: #d1e7dd;
        border-left: 5px solid #198754;
    }
    .node-failed {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    
    /* Node card style */
    .node-card {
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 5px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
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
                
                # Initialize graph view
                st.session_state.graph_view = GraphView()
            
            # Task input and running
            if 'root_node_id' not in st.session_state:
                task_form = st.form(key="task_form")
                with task_form:
                    task_description = st.text_area("Enter Task Description:", height=100)
                    
                    # Add constraints input with examples
                    st.write("Add constraints (optional):")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        add_format_constraint = st.checkbox("Require JSON", key="format_json_checkbox")
                    
                    with col2:
                        add_code_constraint = st.checkbox("Generate code", key="generate_code_checkbox")
                        
                    custom_constraint = st.text_input("Custom constraint:", key="custom_constraint_input")
                    
                    # Only have a single submit button in the form
                    submitted = st.form_submit_button("Run Task")
                
                # Process the form submission outside the form
                if submitted and task_description:
                    # Build constraints list
                    constraints = []
                    if add_format_constraint:
                        constraints.append("format:json")
                    if add_code_constraint:
                        constraints.append("code:true")
                    if custom_constraint:
                        constraints.append(custom_constraint)
                    
                    # Show a spinner while executing
                    with st.spinner("Initializing task..."):
                        # Run agent with constraints
                        st.session_state.agent.run(task_description, constraints if constraints else None)
                    st.experimental_rerun()
            # Display task tree with improved graph view
            else:
                # Add view toggle
                view_type = st.radio("View Type:", ["Task Tree", "Graph View"], horizontal=True)
                
                if view_type == "Graph View":
                    # Use the GraphView component for visualization
                    selected_node_id = st.session_state.get('selected_node_id')
                    
                    try:
                        # Try to render the graph visualization
                        st.session_state.graph_view.render_graph(
                            st.session_state.node_lookup,
                            st.session_state.root_node_id,
                            selected_node_id
                        )
                    except Exception as e:
                        # If graph rendering fails, show a simple tree view instead
                        st.error(f"Error rendering graph: {str(e)}. Using simple tree view instead.")
                        st.session_state.graph_view.render_simple_tree(
                            st.session_state.node_lookup,
                            st.session_state.root_node_id,
                            selected_node_id
                        )
                    
                    # If a node is selected, show its details
                    if selected_node_id and selected_node_id in st.session_state.node_lookup:
                        node = st.session_state.node_lookup[selected_node_id]
                        
                        # Display node details
                        with st.expander(f"Selected Node: {node.retrieve_from_memory('task')}", expanded=True):
                            st.write(f"**Status:** {node.status}")
                            
                            if node.status == STATUS_PENDING:
                                if st.button("Execute Node", key=f"exec_{node.node_id}"):
                                    st.session_state.agent.agentFlow("execute", node)
                                    st.experimental_rerun()
                            
                            # Show output if available
                            if node.output:
                                st.write("**Output:**")
                                st.text(node.output[:500] + ("..." if len(node.output) > 500 else ""))
                                
                                # Process code in output
                                code_files = node.extract_code_files()
                                if code_files:
                                    st.write("**Generated Code Files:**")
                                    for filepath, content in code_files.items():
                                        if st.button(f"Open in Editor: {filepath}", key=f"open_{node.node_id}_{filepath}"):
                                            # Set active file in editor
                                            st.session_state.active_file = filepath
                                            st.session_state.file_content = content
                                            st.experimental_rerun()
                else:  # Task Tree view
                    # Hierarchical tree display
                    st.subheader("Task Hierarchy")
                    
                    # Function to recursively render the tree
                    def render_node_tree(node_id, indent=0):
                        if node_id not in st.session_state.node_lookup:
                            return
                            
                        node = st.session_state.node_lookup[node_id]
                        task = node.retrieve_from_memory("task") or f"Node {node_id[:8]}"
                        
                        # Determine CSS class based on status
                        status_class = f"node-{node.status}"
                        
                        # Status emoji
                        status_emoji = {
                            STATUS_PENDING: "🔘",
                            STATUS_RUNNING: "⏳",
                            STATUS_COMPLETED: "✅",
                            STATUS_FAILED: "❌"
                        }.get(node.status, "⚪")
                        
                        # Node container with appropriate styling
                        st.markdown(f'<div class="node-card {status_class}" style="margin-left: {indent*20}px">', unsafe_allow_html=True)
                        
                        # Node header with task and status
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.write(f"**{status_emoji} {task}**")
                        with col2:
                            if st.button("Select", key=f"select_{node_id}"):
                                st.session_state.selected_node_id = node_id
                                st.session_state.agent.agentFlow("select", node)
                                st.experimental_rerun()
                        
                        # Node children (recursively)
                        for child_id in node.child_ids:
                            render_node_tree(child_id, indent + 1)
                            
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Start rendering from root node
                    render_node_tree(st.session_state.root_node_id)
                
                # Selected node details (common for both views)
                if 'selected_node_id' in st.session_state and st.session_state.selected_node_id:
                    selected_node_id = st.session_state.selected_node_id
                    if selected_node_id in st.session_state.node_lookup:
                        node = st.session_state.node_lookup[selected_node_id]
                        
                        st.markdown("<hr>", unsafe_allow_html=True)
                        st.subheader("Node Details")
                        
                        # Put basic info and actions in a more compact layout
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write(f"**Status:** {node.status}")
                        with col2:
                            st.write(f"**Task:** {node.retrieve_from_memory('task')}")
                        with col3:
                            st.write(f"**Depth:** {node.depth}")
                        
                        # Actions in a single row
                        action_cols = st.columns(3)
                        
                        with action_cols[0]:
                            # Execute button with spinner
                            if node.status == STATUS_PENDING:
                                if st.button("Execute", key=f"execute_detail_{node.node_id}"):
                                    with st.spinner("Executing node..."):
                                        # Use a container to prevent layout shifts
                                        execution_result = st.container()
                                        with execution_result:
                                            st.session_state.agent.agentFlow("execute", node)
                                    st.experimental_rerun()
                        
                        with action_cols[1]:
                            # Regenerate with guidance in a more compact way
                            if node.status in [STATUS_COMPLETED, STATUS_FAILED]:
                                with st.expander("Regenerate with guidance", expanded=False):
                                    regeneration_guidance = st.text_area(
                                        "Guidance:", 
                                        key=f"regen_guidance_{node.node_id}"
                                    )
                                    if st.button("Regenerate", key=f"regenerate_detail_{node.node_id}"):
                                        with st.spinner("Regenerating..."):
                                            st.session_state.agent.agentFlow("regenerate", node, regeneration_guidance)
                                        st.experimental_rerun()
                        
                        with action_cols[2]:
                            # Delete button
                            if st.button("Delete", key=f"delete_detail_{node.node_id}"):
                                st.session_state.agent.agentFlow("delete", node)
                                if node.node_id == st.session_state.selected_node_id:
                                    st.session_state.selected_node_id = None
                                st.experimental_rerun()
                        
                        # Show output using tabs instead of expanders
                        if node.output:
                            # Get code files from node
                            code_files = node.extract_code_files()
                            
                            # Create tab labels
                            tab_labels = ["Output"]
                            if code_files:
                                tab_labels.append(f"Code Files ({len(code_files)})")
                            
                            # Create tabs
                            output_tabs = st.tabs(tab_labels)
                            
                            # Output tab
                            with output_tabs[0]:
                                with st.container():
                                    st.code(node.output)
                            
                            # Code files tab (if any)
                            if code_files and len(output_tabs) > 1:
                                with output_tabs[1]:
                                    # Create file selector
                                    file_paths = list(code_files.keys())
                                    
                                    # Use selectbox for file selection instead of buttons
                                    selected_file_idx = st.selectbox(
                                        "Select file:", 
                                        range(len(file_paths)),
                                        format_func=lambda i: file_paths[i],
                                        key=f"file_select_{node.node_id}"
                                    )
                                    
                                    # Display the selected file
                                    if file_paths:
                                        selected_file = file_paths[selected_file_idx]
                                        content = code_files[selected_file]
                                        language = selected_file.split('.')[-1] if '.' in selected_file else 'text'
                                        st.code(content, language=language)
                                        
                                        # File actions
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            # Use a unique key for the save button
                                            save_key = f"save_code_{node.node_id}_{selected_file}"
                                            if st.button("Save File", key=save_key):
                                                try:
                                                    # Create directory if needed
                                                    directory = os.path.dirname(selected_file)
                                                    if directory:
                                                        os.makedirs(directory, exist_ok=True)
                                                    
                                                    # Save file
                                                    with open(selected_file, 'w', encoding='utf-8') as f:
                                                        f.write(content)
                                                    st.success(f"File saved: {selected_file}")
                                                except Exception as e:
                                                    st.error(f"Error saving file: {str(e)}")
                                        
                                        with col2:
                                            # Use a unique key for the editor button
                                            editor_key = f"open_code_{node.node_id}_{selected_file}"
                                            if st.button("Open in Editor", key=editor_key):
                                                st.session_state.active_file = selected_file
                                                st.session_state.file_content = content
                                                st.experimental_rerun()
                        
                        # Add child node option in a more compact layout
                        if node.status == STATUS_COMPLETED:
                            with st.expander("Add Child Task", expanded=False):
                                new_task = st.text_area("Task description:", key=f"new_task_{node.node_id}")
                                if st.button("Add Task", key=f"add_task_{node.node_id}"):
                                    if new_task.strip():
                                        with st.spinner("Creating child node..."):
                                            child_node = st.session_state.agent.create_child_node(
                                                node, 
                                                new_task,
                                                node.depth + 1
                                            )
                                            st.session_state.selected_node_id = child_node.node_id
                                        st.experimental_rerun()
                                    else:
                                        st.error("Please enter a task description.")
                
                # Session controls
                st.markdown("<hr>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("New Task", key="new_task_btn"):
                        st.session_state.agent.reset_agent()
                        st.session_state.selected_node_id = None
                        st.experimental_rerun()
                
                with col2:
                    if 'show_save_load' not in st.session_state:
                        st.session_state.show_save_load = False
                        
                    if st.button("Save/Load Session", key="save_load_toggle"):
                        st.session_state.show_save_load = not st.session_state.show_save_load
                        st.experimental_rerun()
                
                # Show save/load controls if toggled
                if st.session_state.get('show_save_load', False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        save_filename = st.text_input("Save session as:", "agent_session.json")
                        if st.button("Save"):
                            st.session_state.agent.save_session(save_filename)
                            st.success(f"Session saved to {save_filename}")
                    
                    with col2:
                        load_filename = st.text_input("Load session from:", "agent_session.json")
                        if st.button("Load") and os.path.exists(load_filename):
                            st.session_state.agent.load_session(load_filename)
                            st.success(f"Session loaded from {load_filename}")
                            st.experimental_rerun()
                        elif st.button("Load") and not os.path.exists(load_filename):
                            st.error(f"File not found: {load_filename}")
            
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
            
            # Check if we need to create/update a file from agent output
            if 'active_file' in st.session_state and 'file_content' in st.session_state:
                active_file = st.session_state.active_file
                file_content = st.session_state.file_content
                
                # Create directory if it doesn't exist
                directory = os.path.dirname(active_file)
                if directory and not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                
                # Pre-populate editor with file content
                st.session_state.editor_content = file_content
                
                # Show save button for the generated file
                if st.button("Save Generated File", key=f"save_generated_{active_file}"):
                    try:
                        with open(active_file, 'w', encoding='utf-8') as f:
                            f.write(file_content)
                        st.success(f"File saved: {active_file}")
                    except Exception as e:
                        st.error(f"Error saving file: {str(e)}")
            
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
