# SMARTAGENT/agent/node.py
"""Node class for representing tasks in the agent's hierarchy."""

import streamlit as st
import uuid
import os
from typing import Optional, Dict, Any, List
from .memory import LocalMemory
from .utils import extract_json_from_text
from .constants import STATUS_PENDING, STATUS_RUNNING, STATUS_COMPLETED, STATUS_FAILED, STATUS_OVERRIDDEN

class Node:
    """
    A node represents a discrete task or step in the agent's execution.
    Each node has its own local memory and can access global memory.
    """
    def __init__(self, parent_id: Optional[str] = None, task_description: Optional[str] = None, depth: int = 0):
        """Initialize a new Node.
        
        Args:
            parent_id: ID of the parent node
            task_description: Description of the task for this node
            depth: Depth level in the task tree (0 for root)
        """
        self.node_id = str(uuid.uuid4())
        self.depth = depth
        self.local_memory = LocalMemory(self.node_id)
        self.parent_id = parent_id
        self.child_ids: List[str] = []
        self.status = STATUS_PENDING
        self.output = ""  # Store raw output
        self.error_message = ""
        self.task_description = task_description
        
        if task_description:
            self.store_in_memory("task", task_description)
        
    def store_in_memory(self, key: str, value: Any) -> None:
        """Store data in this node's local memory."""
        self.local_memory.store(key, value)
        
    def retrieve_from_memory(self, key: str) -> Any:
        """Retrieve data from this node's local memory."""
        return self.local_memory.retrieve(key)
    
    def add_child(self, child_id: str) -> None:
        """Add a child node ID to this node."""
        if child_id not in self.child_ids:
            self.child_ids.append(child_id)
    
    def update_status(self, status: str) -> None:
        """Update the status of this node."""
        self.status = status
        self.store_in_memory("status", status)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of this node's state and memory."""
        return {
            "node_id": self.node_id,
            "status": self.status,
            "parent_id": self.parent_id,
            "children": self.child_ids,
            "task": self.retrieve_from_memory("task") or "",
            "result": self.retrieve_from_memory("result") or ""
        }
    
    def build_prompt(self) -> str:
        """Constructs the prompt for the LLM."""
        prompt = []
        # Add global context
        if 'agent' in st.session_state:
            prompt.append(st.session_state.agent.global_memory.get_context())

        # Add task description
        task_description = self.retrieve_from_memory("task")
        if task_description:
            prompt.append(f"Task: {task_description}")

        # Add constraints
        if self.node_id in st.session_state.attention_mechanism.constraints:
            constraints = st.session_state.attention_mechanism.constraints[self.node_id]
            if constraints:
                prompt.append("Constraints:")
                for constraint in constraints:
                    prompt.append(f"- {constraint}")

        # Add parent output if applicable
        if self.parent_id and self.parent_id in st.session_state.node_lookup:
            parent_node = st.session_state.node_lookup[self.parent_id]
            parent_output = parent_node.output
            if parent_output:
                prompt.append(f"Output from Parent Node ({parent_node.node_id[:8]}...): {parent_output}")

        # Add regeneration guidance if applicable
        regeneration_guidance = self.retrieve_from_memory("regeneration_guidance")
        if regeneration_guidance:
            prompt.append(f"Regeneration Guidance: {regeneration_guidance}")

        # Join and return
        return "\n\n".join(prompt)
    
    def process_llm_output(self, llm_output: str) -> None:
        """Processes the raw LLM output, extracting subtasks or results."""
        try:
            parsed_output = extract_json_from_text(llm_output)
            if parsed_output:
                if "subtasks" in parsed_output and isinstance(parsed_output["subtasks"], list):
                    for subtask in parsed_output["subtasks"]:
                        if isinstance(subtask, str):
                            # Simple string subtask
                            self.create_child_node(subtask, self.depth + 1)
                        elif isinstance(subtask, dict) and "task_description" in subtask:
                            # Subtask with description
                            self.create_child_node(subtask["task_description"], self.depth + 1)
                        # else: ignore malformed subtasks
                elif "result" in parsed_output:
                    self.store_in_memory("result", parsed_output["result"])
                else:
                    # Store the entire parsed output if no specific keys are found
                    self.store_in_memory("result", parsed_output)
            else:
                # If no JSON, store the entire output as the result
                self.store_in_memory("result", llm_output)

        except Exception as e:
            self.status = "failed"
            self.error_message = f"Error processing LLM output: {str(e)}"

    def create_child_node(self, task_description: str, depth: int) -> "Node":
        """Creates a child node and adds it to the tree."""
        from . import node as node_module  # Import here to avoid circular imports
        new_node = node_module.Node(parent_id=self.node_id, task_description=task_description, depth=depth)
        st.session_state.node_lookup[new_node.node_id] = new_node
        self.add_child(new_node.node_id)
        st.session_state.attention_mechanism.track_dependencies(self.node_id, new_node.node_id)
        return new_node
    
    def save_state(self, directory: str = "node_memory") -> None:
        """Save the node's state and memory to disk."""
        self.local_memory.save_to_disk(directory)
    
    def load_state(self, directory: str = "node_memory") -> bool:
        """Load the node's state and memory from disk."""
        return self.local_memory.load_from_disk(directory)
