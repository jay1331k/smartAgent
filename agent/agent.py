# SMARTAGENT/agent/agent.py
import streamlit as st
from .attention_mechanism import AttentionMechanism
from .node import Node
from .memory import LocalMemory, GlobalMemory
from typing import Optional
import json
import os
from .utils import handle_retryable_error

# --- Constants --- (Moved outside the class)
MAX_RETRIES: int = 3
RETRY_DELAY: int = 2
LLM_MODEL: str = "gemini-1.5-pro-002"
LLM_TEMPERATURE: float = 0.5
LLM_MAX_TOKENS: int = 1200
GLOBAL_CONTEXT_SUMMARY_INTERVAL: int = 5
MAX_DEPTH: int = 5
# ---


class Agent:
    def __init__(self, llm, llm_config, global_context: str = "This agent decomposes complex tasks.") -> None:
        self.attention_mechanism = AttentionMechanism()
        self.global_memory = GlobalMemory()
        self.global_memory.update_context(global_context)
        self.llm = llm
        self.llm_config = llm_config
        self.execution_count = 0

    def run(self, task_description: str, initial_constraints: Optional[list[str]] = None) -> None:
        """Main entry point for running the agent."""
        self.reset_agent()
        root_node = self.create_root_node(task_description, initial_constraints)
        st.session_state.root_node_id = root_node.node_id
        # No initial execution; wait for user input

    def reset_agent(self) -> None:
        """Resets the agent's state."""
        st.session_state.clear()  # Clear everything
        if os.path.exists("agent_memory.json"):
            os.remove("agent_memory.json")
        self.setup_agent()
        st.session_state.selected_node_id = None  # Reset selected node
        st.session_state.previous_state = None  # Reset Undo

    def setup_agent(self) -> None:
        """Initializes the agent's components."""
        st.session_state.node_lookup = {}
        self.attention_mechanism.add_constraint_checker("format", self.attention_mechanism._check_json_format)
        self.attention_mechanism.add_constraint_checker("contains", self.attention_mechanism._check_contains_word)
        self.attention_mechanism.add_constraint_checker("max_length", self.attention_mechanism._check_max_length)
        st.session_state.attention_mechanism = self.attention_mechanism
        st.session_state.agent = self
        st.session_state.llm = self.llm
        st.session_state.llm_config = self.llm_config

    def create_root_node(self, task_description: str, initial_constraints: Optional[list[str]] = None) -> Node:
        """Creates the root node."""
        new_node = Node(parent_id=None, task_description=task_description, depth=0)
        st.session_state.node_lookup[new_node.node_id] = new_node
        if initial_constraints:
            for constraint in initial_constraints:
                st.session_state.attention_mechanism.add_constraint(new_node.node_id, constraint)
        st.session_state.attention_mechanism.track_dependencies(None, new_node.node_id)
        return new_node

    def create_child_node(self, parent_node: Node, task_description: str, depth: int) -> Node:
        """Creates a child node."""
        new_node = Node(parent_id=parent_node.node_id, task_description=task_description, depth=depth)
        st.session_state.node_lookup[new_node.node_id] = new_node
        parent_node.add_child(new_node)
        st.session_state.attention_mechanism.track_dependencies(parent_node.node_id, new_node.node_id)
        return new_node

    def delete_node_and_children(self, node: Node) -> None:
        """Recursively deletes a node and all its descendants."""

        # Remove children first
        for child_id in node.child_ids:
            if child_id in st.session_state.node_lookup:
                self.delete_node_and_children(st.session_state.node_lookup[child_id])

        # Now remove the node itself
        st.session_state.node_lookup.pop(node.node_id, None)
        st.session_state.attention_mechanism.remove_node(node.node_id)

    def agentFlow(self, action: str, node: Node, regeneration_guidance: str = "") -> None:
        """
        Centralized function to handle node actions (execution, regeneration, deletion).
        This improves code organization and makes the main UI logic cleaner.
        """
        if action == "execute":
            self._execute_node(node)
        elif action == "regenerate":
            self._regenerate_node(node, regeneration_guidance)
        elif action == "delete":
            self.delete_node_and_children(node)
        else:
            raise ValueError(f"Invalid action: {action}")

    def _execute_node(self, node: Node) -> None:
        """Executes a single node."""
        node.status = "running"
        prompt = node.build_prompt()

        for attempt in range(MAX_RETRIES):  # Use the constant here
            try:
                response = self.llm.generate_content(
                    prompt,
                    generation_config=st.session_state.llm_config
                )
                llm_output = response.text
                try:
                    json.loads(llm_output)  # Basic JSON validation
                except json.JSONDecodeError:
                    raise ValueError(f"LLM output is not valid JSON: {llm_output}")

                node.output = llm_output
                node.process_llm_output(llm_output)

                if not st.session_state.attention_mechanism.check_constraints(node):
                    return  # Constraint check failed
                break  # Exit loop if successful

            except Exception as e:
                if handle_retryable_error(node, attempt, e):
                    return

        if node.status == "running":  # If still in running state
            node.status = "completed"
            if not node.child_ids:  # Leaf node
                st.session_state.attention_mechanism.summarize_node(node)

    def _regenerate_node(self, node: Node, regeneration_guidance: str) -> None:
        """Resets a node for regeneration and stores guidance."""
        node.status = "pending"
        node.output = ""
        node.error_message = ""
        node.store_in_memory("regeneration_guidance", regeneration_guidance)

    def save_session(self, filename: str) -> None:
        """Saves the current session (node_lookup and attention_mechanism) to a JSON file."""
        data = {
            "node_lookup": {node_id: node.__dict__ for node_id, node in st.session_state.node_lookup.items()},
            "attention_mechanism": {
                "dependency_graph": st.session_state.attention_mechanism.dependency_graph,
                "constraints": st.session_state.attention_mechanism.constraints,
            },
            "root_node_id": st.session_state.root_node_id,
            "global_memory": st.session_state.agent.global_memory.global_context,  # Save Global Memory
            "execution_count": st.session_state.agent.execution_count  # Save Execution Count
        }

        # Convert Node objects to dictionaries (and LocalMemory)
        for node_id, node_data in data["node_lookup"].items():
            node_data["local_memory"] = node_data["local_memory"].local_memory

        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

    def load_session(self, filename: str) -> None:
        """Loads a session from a JSON file."""
        with open(filename, "r") as f:
            data = json.load(f)

        # Clear existing state
        self.reset_agent()

        st.session_state.root_node_id = data["root_node_id"]
        # Re-create Node objects and LocalMemory
        st.session_state.node_lookup = {}
        for node_id, node_data in data["node_lookup"].items():
            new_node = Node(node_data["parent_id"], node_data["task_description"], node_data["depth"])
            new_node.node_id = node_data["node_id"]
            new_node.child_ids = node_data["child_ids"]
            new_node.status = node_data["status"]
            new_node.output = node_data["output"]
            new_node.error_message = node_data["error_message"]
            new_node.local_memory.local_memory = node_data["local_memory"]  # Load the dictionary
            st.session_state.node_lookup[node_id] = new_node

        st.session_state.attention_mechanism.dependency_graph = data["attention_mechanism"]["dependency_graph"]
        st.session_state.attention_mechanism.constraints = data["attention_mechanism"]["constraints"]
        st.session_state.agent.global_memory.update_context(data["global_memory"])  # Load Global Memory
        st.session_state.agent.execution_count = data["execution_count"]  # Load Execution Count