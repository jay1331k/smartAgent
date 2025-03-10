# SMARTAGENT/agent/utils.py
import time
import re
import streamlit as st
import graphviz
import plotly.graph_objects as go
#NEW: Import constants directly from the agent module.
from agent import agent


def parse_constraint(constraint_string: str) -> tuple[str, str]:
    """Parses a constraint string into its type and value."""
    try:
        constraint_type, constraint_value = constraint_string.split(":", 1)
        return constraint_type.strip(), constraint_value.strip()
    except ValueError:
        return "unknown", constraint_string  # Default type if parsing fails

def handle_retryable_error(node: "Node", attempt: int, error: Exception) -> bool:
    """Handles retryable LLM API errors (rate limits, timeouts)."""
    if "429" in str(error) or "rate limit" in str(error).lower() or "timeout" in str(error).lower():
        if attempt < agent.MAX_RETRIES - 1:  # Use Agent's constants
            delay = agent.RETRY_DELAY * (2 ** attempt)
            st.toast(f"{type(error).__name__}. Retrying in {delay} seconds (attempt {attempt + 2}/{agent.MAX_RETRIES})...")
            time.sleep(delay)
            return False
        else:
            node.status = "failed"
            node.error_message = f"LLM API Error: Max retries exceeded: {error}"
            st.error(node.error_message)
            return True
    else:
        node.status = "failed"
        node.error_message = f"LLM API Error: {error}"
        st.error(node.error_message)
        return True  # Stop retrying (non-retryable error)

def generate_plotly_graph() -> go.Figure:
    """Generates an interactive Plotly graph of the task decomposition tree."""
    nodes = st.session_state.node_lookup
    edges = []
    node_labels = {}
    node_colors = []
    # --- Highlighting Logic ---
    highlighted_nodes = set()
    if st.session_state.selected_node_id:
        selected_node = nodes.get(st.session_state.selected_node_id)
        if selected_node:
            highlighted_nodes.add(selected_node.node_id)
            if selected_node.parent_id:
                highlighted_nodes.add(selected_node.parent_id)
            highlighted_nodes.update(selected_node.child_ids)
    # ---

    for node_id, node in nodes.items():
        # --- Tooltips ---
        node_labels[node_id] = (
            f"ID: {node.node_id}<br>"
            f"Task: {node.task_description}<br>"  # Full description here
            f"Status: {node.status}<br>"
            f"Output Snippet: {node.output[:50]}..." if node.output else "No output yet"  # Snippet
        )
        # ---

        if node.status == "failed":
            color = "red"
        elif node.status == "completed":
            color = "green"
        elif node.status == "overridden":
            color = "orange"
        elif node.status == "running":  # Distinct color for "running"
            color = "blue"
        elif node.status == "pending": # Distinct color for "pending"
            color = 'purple'
        else:
            color = "gray"

        # Apply highlighting
        if node_id in highlighted_nodes:
            node_colors.append(color)  # Use the determined color
        else:
            node_colors.append("lightgray")  # Dim non-highlighted nodes


        if node.parent_id:
            edges.append((node.parent_id, node.node_id))

    layout = graphviz.Digraph()
    for node_id in nodes:
        layout.node(node_id)
    for edge in edges:
        layout.edge(edge[0], edge[1])
    pos = layout.pipe(format='json').decode('utf-8')
    pos = eval(pos)

    node_positions = {node['name']: (float(node['pos'].split(',')[0]), float(node['pos'].split(',')[1])) for node in pos['objects']}

    edge_x = []
    edge_y = []
    for edge in edges:
        x0, y0 = node_positions[edge[0]]
        x1, y1 = node_positions[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x = []
    node_y = []
    node_text = []
    for node_id in nodes:
        x, y = node_positions[node_id]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node_labels[node_id])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',  # Use 'text' for tooltips
        text=node_text,
        marker=dict(
            showscale=False,
            color=node_colors,
            size=10,
            line_width=2))

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=0),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    fig.update_layout(clickmode='event+select')
    fig.data[1].on_click(handle_node_click)

    return fig

def handle_node_click(trace, points, state):
    """Handles node click events in the Plotly graph."""
    if points.point_inds:
        clicked_node_id = list(st.session_state.node_lookup.keys())[points.point_inds[0]]
        st.session_state.selected_node_id = clicked_node_id
        st.rerun()

def display_node_textual(node_id: str, selected_node_id: str = None, level: int = 0) -> None:
    """Displays the node hierarchy in a textual format."""
    if node_id not in st.session_state.node_lookup:
        return
    node = st.session_state.node_lookup[node_id]
    indent = "    " * level
    # Highlight selected node and indicate status more clearly
    if node_id == selected_node_id:
        st.write(f"{indent}➡️ **Node ID:** {node.node_id} (**Status:** {node.status})")
    else:
        st.write(f"{indent}- **Node ID:** {node.node_id} (**Status:** {node.status})")
    st.write(f"{indent}  **Task:** {node.task_description}")
    if node.child_ids:
        for child_id in node.child_ids:
            display_node_textual(child_id, selected_node_id, level + 1)

def display_selected_node_details(node_id: str) -> None:
    """Displays detailed information for the selected node, with action buttons."""
    if node_id not in st.session_state.node_lookup:
        return

    node = st.session_state.node_lookup[node_id]

    st.write(f"### Details for Node: {node.node_id}")
    st.write(f"**Task:** {node.task_description}")
    st.write(f"**Status:** {node.status}")  # Status is already clear
    st.write("**Constraints:**")
    for constraint in st.session_state.attention_mechanism.get_constraints(node_id):
        st.write(f"- {constraint}")

    st.write("**Dependencies:**")
    for dep_id in st.session_state.attention_mechanism.get_dependencies(node_id):
        dep_node = st.session_state.node_lookup.get(dep_id)
        if dep_node:
            st.write(f"- Depends on: {dep_node.node_id} ({dep_node.task_description})")
        else:
            st.write(f"- Depends on: {dep_id} (NOT FOUND - check dependency graph!)")

    st.write("**Output:**")
    st.write(node.output)
    if node.error_message:
        st.write(f"**Error:** {node.error_message}")  # Error message display
    with st.expander("Local Memory"):
        for key, value in node.local_memory.items():
            st.write(f"*   **{key}:** {value}")

    # --- Action Buttons (with visual cue) ---
    # Disable buttons based on status, and use type="primary" for Execute
    if node.status == "pending":
        if st.button("Execute Node", key=f"execute_{node_id}", type="primary"):  # Primary button
            st.session_state.previous_state = {
                "node_lookup": st.session_state.node_lookup.copy(),
                "attention_mechanism": st.session_state.attention_mechanism,
                "global_memory": st.session_state.agent.global_memory.get_context(),  # Save Global Memory
                "execution_count": st.session_state.agent.execution_count  # Save Execution Count
            }
            # node.execute() # OLD - Replaced with agentFlow call
            st.session_state.agent.agentFlow("execute", node)
            st.rerun()

    if node.status in ("running", "completed", "failed", "overridden"):
        with st.form(key=f"regenerate_form_{node_id}"):
            regeneration_guidance = st.text_input("Enter regeneration guidance (optional):", key=f"guidance_{node_id}")
            if st.form_submit_button("Regenerate Node", key=f"regenerate_{node_id}"):
                st.session_state.previous_state = {
                    "node_lookup": st.session_state.node_lookup.copy(),
                    "attention_mechanism": st.session_state.attention_mechanism,
                    "global_memory": st.session_state.agent.global_memory.get_context(),  # Save Global Memory
                    "execution_count": st.session_state.agent.execution_count  # Save Count

                }
                # node.status = "pending" # OLD - Replaced with agentFlow
                # node.output = ""
                # node.error_message = ""
                # node.store_in_memory("regeneration_guidance", regeneration_guidance)
                st.session_state.agent.agentFlow("regenerate", node, regeneration_guidance)
                st.rerun()

    if st.button("Delete Node", key=f"delete_{node_id}"):
        # --- Confirmation Dialog ---
        if st.session_state.get(f"confirm_delete_{node_id}", False):  # Check for confirmation
            st.session_state.previous_state = {
                "node_lookup": st.session_state.node_lookup.copy(),
                "attention_mechanism": st.session_state.attention_mechanism,
                "global_memory": st.session_state.agent.global_memory.get_context(),  # Save Global Memory
                "execution_count": st.session_state.agent.execution_count  # Save Count
            }
            st.session_state.agent.agentFlow("delete", node)
            st.session_state.selected_node_id = None
            st.session_state[f"confirm_delete_{node_id}"] = False  # Reset confirmation
            st.rerun()
        else:
            st.warning(f"Are you sure you want to delete Node {node_id} and all its children?")
            if st.button("Confirm Delete", key=f"confirm_{node_id}"):
                st.session_state[f"confirm_delete_{node_id}"] = True
                st.rerun()

    # ---