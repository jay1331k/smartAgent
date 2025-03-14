import streamlit as st
from components.utils import STATUS_PENDING, STATUS_RUNNING, STATUS_COMPLETED, STATUS_FAILED

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
            
        # Show node output/result - Using direct text display to avoid nested expanders
        if node.output:
            st.write("**Raw Output:**")
            st.code(node.output, language="text")
                
        result = node.retrieve_from_memory("result")
        if result:
            st.write("**Result:**")
            st.write(result)
            
        # Node actions
        col1, col2, col3 = st.columns(3)
        with col1:
            if node.status in [STATUS_PENDING, STATUS_FAILED]:
                if st.button("Execute", key=f"execute_{node_id}"):
                    st.session_state.agent.agentFlow("execute", node)
                    st.experimental_rerun()
        with col2:
            if st.button("Regenerate", key=f"regenerate_{node_id}"):
                guidance = st.text_area("Regeneration Guidance:", key=f"guidance_{node_id}")
                if st.button("Confirm Regeneration", key=f"confirm_regen_{node_id}"):
                    st.session_state.agent.agentFlow("regenerate", node, guidance)
                    st.experimental_rerun()
        with col3:
            if st.button("Delete", key=f"delete_{node_id}"):
                st.session_state.agent.agentFlow("delete", node)
                if st.session_state.get('selected_node_id') == node_id:
                    st.session_state.selected_node_id = None
                st.experimental_rerun()
                
        # Constraints
        st.subheader("Constraints")
        if hasattr(st.session_state, 'attention_mechanism'):
            constraints = st.session_state.attention_mechanism.get_constraints(node_id)
            for i, constraint in enumerate(constraints):
                st.text_input(f"Constraint {i+1}", value=constraint, key=f"constraint_{node_id}_{i}")
                if st.button("Update", key=f"update_constraint_{node_id}_{i}"):
                    new_constraint = st.session_state[f"constraint_{node_id}_{i}"]
                    st.session_state.attention_mechanism.update_constraint(node_id, i, new_constraint)
                    st.experimental_rerun()
                if st.button("Remove", key=f"remove_constraint_{node_id}_{i}"):
                    st.session_state.attention_mechanism.remove_constraint(node_id, i)
                    st.experimental_rerun()
            
            new_constraint = st.text_input("New Constraint", key=f"new_constraint_{node_id}")
            if st.button("Add", key=f"add_constraint_{node_id}"):
                if new_constraint:
                    st.session_state.attention_mechanism.add_constraint(node_id, new_constraint)
                    st.session_state[f"new_constraint_{node_id}"] = ""
                    st.experimental_rerun()
    
    # Render child nodes - OUTSIDE of the parent's expander
    if node.child_ids:
        st.write(f"**Subtasks of {task[:20]}{'...' if len(task) > 20 else ''}:**")
        for child_id in node.child_ids:
            render_node_tree(child_id, depth + 1)
    elif node.status == STATUS_COMPLETED and depth > 0:
        st.write(f"*No subtasks created for this node.*")
