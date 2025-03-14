import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
import io
import base64

from components.utils import STATUS_PENDING, STATUS_RUNNING, STATUS_COMPLETED, STATUS_FAILED

class GraphView:
    """Provides visualization for the agent's execution graph."""
    
    def __init__(self):
        self.status_colors = {
            STATUS_PENDING: "lightgrey",
            STATUS_RUNNING: "lightyellow",
            STATUS_COMPLETED: "lightgreen",
            STATUS_FAILED: "lightcoral",
        }
        self.node_size = 2000
        self.font_size = 8
        
    def build_graph(self, node_lookup: Dict[str, Any], root_node_id: str) -> nx.DiGraph:
        """Build a NetworkX directed graph from the node lookup dictionary."""
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for node_id, node in node_lookup.items():
            label = node.retrieve_from_memory("task")
            if not label:
                label = f"Node {node_id[:6]}..."
                
            # Truncate label if too long
            if len(label) > 30:
                label = label[:27] + "..."
                
            G.add_node(
                node_id, 
                label=label,
                status=node.status,
                depth=node.depth
            )
        
        # Add edges
        for node_id, node in node_lookup.items():
            if node.parent_id and node.parent_id in node_lookup:
                G.add_edge(node.parent_id, node_id)
        
        return G
        
    def render_graph(self, node_lookup: Dict[str, Any], root_node_id: str, selected_node_id: Optional[str] = None) -> None:
        """Render the graph visualization in Streamlit."""
        if not node_lookup or root_node_id not in node_lookup:
            st.warning("No nodes to display in the graph.")
            return
            
        G = self.build_graph(node_lookup, root_node_id)
        
        # Get node positions using hierarchical layout
        try:
            # First try to use graphviz layout
            import pygraphviz
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        except (ImportError, Exception) as e:
            st.info("Using alternative graph layout since pygraphviz is not available. For better hierarchical layouts, install pygraphviz.")
            # Fallback layout options
            if hasattr(nx, "planar_layout"):
                try:
                    pos = nx.planar_layout(G)
                except:
                    # If planar layout fails, try spring layout
                    pos = nx.spring_layout(G)
            else:
                # Tree-like layout for hierarchical data
                try:
                    # Create a BFS tree from the root node
                    T = nx.bfs_tree(G, root_node_id)
                    pos = nx.drawing.nx_pydot.graphviz_layout(T, prog="dot")
                except:
                    # Final fallback to spring layout
                    pos = nx.spring_layout(G, k=0.5, iterations=100)
        
        # Create figure and axis
        plt.figure(figsize=(10, 8))
        
        # Get node colors based on status
        node_colors = [self.status_colors.get(G.nodes[n]['status'], "white") for n in G.nodes()]
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, 
            node_color=node_colors,
            node_size=self.node_size,
            edgecolors='black',
            linewidths=1,
            alpha=0.8
        )
        
        # Highlight selected node if provided
        if selected_node_id and selected_node_id in G:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=[selected_node_id],
                node_color='lightblue',
                node_size=self.node_size,
                edgecolors='blue',
                linewidths=2
            )
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos, 
            arrows=True,
            arrowstyle='-|>',
            width=1.5,
            edge_color='gray'
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            G, pos,
            labels={n: G.nodes[n]['label'] for n in G.nodes()},
            font_size=self.font_size,
            font_weight='bold'
        )
        
        plt.axis('off')
        plt.tight_layout()
        
        # Save the figure to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        
        # Display the figure
        st.image(buf, caption="Task Hierarchy Graph", use_column_width=True)
        plt.close()
        
        # Create clickable areas for node selection
        st.write("**Click on a node to select it:**")
        cols = st.columns(3)
        i = 0
        for node_id in G.nodes():
            node = node_lookup[node_id]
            label = node.retrieve_from_memory("task")
            if not label:
                label = f"Node {node_id[:6]}..."
                
            if len(label) > 20:
                label = label[:17] + "..."
                
            status_color = {
                STATUS_PENDING: "🔘",
                STATUS_RUNNING: "⏳",
                STATUS_COMPLETED: "✅",
                STATUS_FAILED: "❌",
            }.get(node.status, "⚪")
            
            with cols[i % 3]:
                if st.button(f"{status_color} {label}", key=f"graph_node_{node_id}"):
                    st.session_state.selected_node_id = node_id
                    st.experimental_rerun()
            
            i += 1
            
    def render_simple_tree(self, node_lookup: Dict[str, Any], root_node_id: str, selected_node_id: Optional[str] = None) -> None:
        """Render a simple tree visualization when graph libraries are not available."""
        if root_node_id not in node_lookup:
            st.warning("Root node not found.")
            return
        
        def _render_node(node_id, depth=0):
            if node_id not in node_lookup:
                return
            
            node = node_lookup[node_id]
            label = node.retrieve_from_memory("task") or f"Node {node_id[:6]}..."
            
            # Determine status indicator
            status_indicator = {
                STATUS_PENDING: "🔘",
                STATUS_RUNNING: "⏳",
                STATUS_COMPLETED: "✅", 
                STATUS_FAILED: "❌"
            }.get(node.status, "⚪")
            
            # Highlight if selected
            is_selected = (node_id == selected_node_id)
            prefix = "  " * depth + ("└─ " if depth > 0 else "")
            
            # Create button for each node
            label_text = f"{prefix}{status_indicator} {label}"
            button_style = "primary" if is_selected else "secondary"
            
            if st.button(label_text, key=f"tree_node_{node_id}", type=button_style):
                st.session_state.selected_node_id = node_id
                st.experimental_rerun()
            
            # Process children
            for child_id in node.child_ids:
                _render_node(child_id, depth + 1)
        
        # Render from root
        st.write("### Tree View (Alternative to Graph)")
        _render_node(root_node_id)
