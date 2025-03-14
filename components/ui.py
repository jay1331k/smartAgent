import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
import io

from components.utils import STATUS_PENDING, STATUS_RUNNING, STATUS_COMPLETED, STATUS_FAILED
import uuid

def render_node_tree(root_node_id: str, expanded_nodes: Optional[Dict[str, bool]] = None):
    """Renders the task tree hierarchically using components in the UI."""
    if expanded_nodes is None:
        expanded_nodes = st.session_state.get("expanded_nodes", {})
        
    if root_node_id not in st.session_state.node_lookup:
        st.warning("Root node not found.")
        return
        
    # Function to render node and its children recursively
    def render_node(node_id: str, depth: int = 0):
        if node_id not in st.session_state.node_lookup:
            return
            
        node = st.session_state.node_lookup[node_id]
        task_description = node.retrieve_from_memory("task") or f"Node {node_id[:8]}..."
        
        # Status indicator
        status_indicator = {
            STATUS_PENDING: "🔘",
            STATUS_RUNNING: "⏳",
            STATUS_COMPLETED: "✅",
            STATUS_FAILED: "❌"
        }.get(node.status, "⚪")
        
        # Indentation for tree structure
        indent = "  " * depth
        
        # Create a unique key for this node
        node_key = f"node_{node_id}"
        
        # Create expandable section for the node
        is_expanded = expanded_nodes.get(node_id, depth < 1)  # Auto-expand first level
        
        # Node header with expansion toggle and status
        col1, col2, col3 = st.columns([0.7, 8, 1.3])
        
        with col1:
            st.write(f"{status_indicator}")
            
        with col2:
            if node.child_ids:
                if st.button("📂" if is_expanded else "📁", key=f"expand_{node_id}", help="Expand/Collapse"):
                    expanded_nodes[node_id] = not is_expanded
                    st.session_state.expanded_nodes = expanded_nodes
                    st.experimental_rerun()
            else:
                st.write("📄")
                
            st.write(f"{task_description}")
            
        with col3:
            if st.button("Select", key=f"select_{node_id}"):
                st.session_state.selected_node_id = node_id
                st.experimental_rerun()
        
        # If expanded and has children, show them
        if is_expanded and node.child_ids:
            for child_id in node.child_ids:
                render_node(child_id, depth + 1)
    
    # Start rendering from the root
    render_node(root_node_id)

def render_node_graph(root_node_id: str):
    """Renders the task tree as a network graph."""
    if root_node_id not in st.session_state.node_lookup:
        st.warning("Root node not found.")
        return
        
    # Create graph
    G = nx.DiGraph()
    
    # Add nodes and edges
    for node_id, node in st.session_state.node_lookup.items():
        # Get task description as node label
        label = node.retrieve_from_memory("task") or f"Node {node_id[:8]}..."
        if len(label) > 20:
            label = label[:17] + "..."
            
        # Add node with attributes
        G.add_node(node_id, 
                  label=label,
                  status=node.status)
        
        # Add edge from parent to this node
        if node.parent_id and node.parent_id in st.session_state.node_lookup:
            G.add_edge(node.parent_id, node_id)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Get hierarchical layout
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except ImportError:
        # Fallback if graphviz is not available
        pos = nx.spring_layout(G)
    
    # Node colors based on status
    node_colors = []
    for node in G.nodes():
        status = G.nodes[node]["status"]
        color = {
            STATUS_PENDING: "lightgray",
            STATUS_RUNNING: "yellow",
            STATUS_COMPLETED: "lightgreen",
            STATUS_FAILED: "lightcoral"
        }.get(status, "white")
        node_colors.append(color)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, 
                          node_size=1500, 
                          node_color=node_colors,
                          edgecolors="black")
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, 
                          arrows=True,
                          arrowsize=20,
                          width=1.5)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, 
                           labels={n: G.nodes[n]["label"] for n in G.nodes()},
                           font_size=8)
    
    # Save to buffer and display in Streamlit
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    
    # Display image
    st.image(buf, use_column_width=True)
    plt.close()
    
    # Create clickable buttons for nodes
    st.write("**Click on a node to select it:**")
    
    cols = st.columns(3)
    i = 0
    
    # Sort nodes by depth for better display
    nodes_by_depth = {}
    for node_id in G.nodes():
        node = st.session_state.node_lookup[node_id]
        depth = node.depth
        if depth not in nodes_by_depth:
            nodes_by_depth[depth] = []
        nodes_by_depth[depth].append(node_id)
    
    # Display nodes by depth
    for depth in sorted(nodes_by_depth.keys()):
        st.write(f"**Level {depth}:**")
        
        for node_id in nodes_by_depth[depth]:
            node = st.session_state.node_lookup[node_id]
            task = node.retrieve_from_memory("task") or f"Node {node_id[:8]}..."
            if len(task) > 25:
                task = task[:22] + "..."
                
            status_icon = {
                STATUS_PENDING: "🔘",
                STATUS_RUNNING: "⏳",
                STATUS_COMPLETED: "✅",
                STATUS_FAILED: "❌"
            }.get(node.status, "⚪")
            
            with cols[i % 3]:
                if st.button(f"{status_icon} {task}", key=f"graph_node_{node_id}"):
                    st.session_state.selected_node_id = node_id
                    st.experimental_rerun()
            
            i += 1
``` 
