import streamlit as st
from components.utils import STATUS_PENDING, STATUS_RUNNING, STATUS_COMPLETED, STATUS_FAILED
import uuid

def render_node_tree(node_id, depth=0):
    if not node_id or node_id not in st.session_state.node_lookup:
        return

    node = st.session_state.node_lookup[node_id]
    
    # Create an expander for this node
    # Fix the task retrieval by checking task_description property first
    task = node.task_description or node.retrieve_from_memory("task") or "Task not specified"
    
    status_emoji = {
        STATUS_PENDING: "🔄",
        STATUS_RUNNING: "⏳",
        STATUS_COMPLETED: "✅",
        STATUS_FAILED: "❌"
    }.get(node.status, "❓")
    
    expander_label = f"{status_emoji} {task[:50]}{'...' if len(task) > 50 else ''}"
    is_selected = node_id == st.session_state.get('selected_node_id')
    
    with st.expander(expander_label, expanded=is_selected or depth == 0):
        # Node selection button
        if st.button(f"Select Node", key=f"select_{node_id}"):
            st.session_state.selected_node_id = node_id
            st.experimental_rerun()
            
        # Node content
        st.write(f"**Status:** {node.status.capitalize()}")
        if node.status == STATUS_FAILED and node.error_message:
            st.error(node.error_message)
            
        # Show node output/result - FIXED: Using simpler approach without modifying session state
        if node.output:
            # Create a unique key for the checkbox
            output_key = f"show_output_{node_id}"
            
            # Just use the checkbox's return value directly without trying to update session state after
            show_output = st.checkbox("📄 Show Raw Output", key=output_key)
            
            # Only display output if checkbox is checked
            if show_output:
                st.code(node.output, language="text")
                
        result = node.retrieve_from_memory("result")
        if result:
            st.write("**Result:**")
            st.write(result)
            
        # Node actions
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        with col1:
            if node.status in [STATUS_PENDING, STATUS_FAILED]:
                if st.button("Execute", key=f"execute_{node_id}"):
                    st.session_state.agent.agentFlow("execute", node)
                    st.experimental_rerun()
        with col2:
            if st.button("Regenerate", key=f"regenerate_{node_id}"):
                # Use a different session state key for the regeneration form toggle
                st.session_state[f"show_regen_form_{node_id}"] = True
                st.experimental_rerun()
                
        # Show regeneration guidance input conditionally
        if st.session_state.get(f"show_regen_form_{node_id}", False):
            guidance = st.text_area("Regeneration Guidance:", key=f"guidance_{node_id}")
            if st.button("Confirm Regeneration", key=f"confirm_regen_{node_id}"):
                st.session_state.agent.agentFlow("regenerate", node, guidance)
                st.session_state[f"show_regen_form_{node_id}"] = False
                st.experimental_rerun()
            if st.button("Cancel", key=f"cancel_regen_{node_id}"):
                st.session_state[f"show_regen_form_{node_id}"] = False
                st.experimental_rerun()
                
        with col3:
            if st.button("Delete", key=f"delete_{node_id}"):
                st.session_state.agent.agentFlow("delete", node)
                if st.session_state.get('selected_node_id') == node_id:
                    st.session_state.selected_node_id = None
                st.experimental_rerun()
        
        with col4:
            # Add visualization button
            if st.button("View Graph", key=f"view_graph_{node_id}"):
                st.session_state['show_graph'] = True
                st.session_state['graph_root_id'] = node_id
                st.experimental_rerun()
                
        # Constraints - FIXED: Use direct display instead of nested expander
        st.subheader("Constraints")
        if hasattr(st.session_state, 'attention_mechanism'):
            constraints = st.session_state.attention_mechanism.get_constraints(node_id)
            
            # Display existing constraints
            for i, constraint in enumerate(constraints):
                # Create columns for constraint and actions
                c1, c2, c3 = st.columns([3, 1, 1])
                with c1:
                    # Use a consistent key for the constraint field but don't modify session state after creation
                    constraint_key = f"constraint_{node_id}_{i}"
                    new_constraint_value = st.text_input(f"Constraint {i+1}", value=constraint, key=constraint_key)
                
                with c2:
                    if st.button("Update", key=f"update_constraint_{node_id}_{i}"):
                        st.session_state.attention_mechanism.update_constraint(node_id, i, new_constraint_value)
                        st.experimental_rerun()
                
                with c3:
                    if st.button("Remove", key=f"remove_constraint_{node_id}_{i}"):
                        st.session_state.attention_mechanism.remove_constraint(node_id, i)
                        st.experimental_rerun()
            
            # Add new constraint - use form to avoid session state modification issues
            with st.form(key=f"new_constraint_form_{node_id}"):
                new_constraint = st.text_input("New Constraint", key=f"new_constraint_input_{node_id}")
                submitted = st.form_submit_button("Add Constraint")
                if submitted and new_constraint:
                    st.session_state.attention_mechanism.add_constraint(node_id, new_constraint)
                    st.experimental_rerun()
    
    # Render child nodes - OUTSIDE of the parent's expander
    if node.child_ids:
        st.write(f"**Subtasks of {task[:20]}{'...' if len(task) > 20 else ''}:**")
        for child_id in node.child_ids:
            render_node_tree(child_id, depth + 1)
    elif node.status == STATUS_COMPLETED and depth > 0:
        st.write(f"*No subtasks created for this node.*")

def render_node_graph(root_node_id):
    """Render a visual graph of the node tree."""
    try:
        import streamlit_agraph as stag
        import networkx as nx
        import matplotlib.pyplot as plt
        from streamlit_agraph import Node as GraphNode, Edge, Config, agraph
        
        # Create nodes and edges
        nodes = []
        edges = []
        node_lookup = st.session_state.node_lookup
        
        # Function to recursively add nodes to the graph
        def add_node_and_children(node_id, is_root=False):
            if node_id not in node_lookup:
                return
            
            node = node_lookup[node_id]
            task = node.task_description or node.retrieve_from_memory("task") or "Task not specified"
            
            # Choose color based on status
            color_map = {
                STATUS_PENDING: "#FFC107",   # Yellow
                STATUS_RUNNING: "#2196F3",   # Blue
                STATUS_COMPLETED: "#4CAF50", # Green
                STATUS_FAILED: "#F44336",    # Red
            }
            color = color_map.get(node.status, "#9E9E9E")  # Default gray
            
            # Create agraph node
            nodes.append(
                GraphNode(
                    id=node_id,
                    label=task[:20] + ("..." if len(task) > 20 else ""),
                    color=color,
                    size=20 if is_root else 15
                )
            )
            
            # Add edges from this node to all its children
            for child_id in node.child_ids:
                edges.append(Edge(source=node_id, target=child_id))
                add_node_and_children(child_id)
        
        # Start with root node
        add_node_and_children(root_node_id, is_root=True)
        
        # Configure and display the graph
        config = Config(
            height=500,
            width=700,
            directed=True,
            physics=True,
            hierarchical=True
        )
        
        st.markdown("### Node Graph Visualization")
        if st.button("Close Graph View"):
            st.session_state['show_graph'] = False
            st.experimental_rerun()
            
        agraph(nodes=nodes, edges=edges, config=config)
        
    except ImportError:
        st.error("Could not render graph. Please install streamlit-agraph: pip install streamlit-agraph")
