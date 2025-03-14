import streamlit as st
import os
import json
import time
from typing import Optional, Dict, List, Any

from components.memory import GlobalMemory, LocalMemory
from components.attention_mechanism import AttentionMechanism
from components.node import Node
from components.utils import handle_node_retryable_error, MAX_DEPTH, MAX_RETRIES, RETRY_DELAY, GLOBAL_CONTEXT_SUMMARY_INTERVAL, STATUS_PENDING, STATUS_RUNNING, STATUS_COMPLETED, STATUS_FAILED

class Agent:
    def __init__(self, llm, llm_config, global_context: str = "This agent decomposes complex tasks.") -> None:
        self.attention_mechanism = AttentionMechanism()
        self.global_memory = GlobalMemory()
        self.global_memory.update_context(global_context)
        self.llm = llm
        self.llm_config = llm_config
        self.execution_count = 0
        self.max_depth = MAX_DEPTH
        self.global_context_summary_interval = GLOBAL_CONTEXT_SUMMARY_INTERVAL

    def run(self, task_description: str, initial_constraints: Optional[list[str]] = None) -> None:
        self.reset_agent()
        root_node = self.create_root_node(task_description, initial_constraints)
        st.session_state.root_node_id = root_node.node_id

    def reset_agent(self) -> None:
        st.session_state.node_lookup = {}
        if 'root_node_id' in st.session_state:
            del st.session_state.root_node_id
        if os.path.exists("agent_memory.json"):
            os.remove("agent_memory.json")
        self.setup_agent()
        st.session_state.selected_node_id = None
        st.session_state.previous_state = None

    def setup_agent(self) -> None:
        st.session_state.node_lookup = {}
        self.attention_mechanism.add_constraint_checker("format", self.attention_mechanism._check_json_format)
        self.attention_mechanism.add_constraint_checker("contains", self.attention_mechanism._check_contains_word)
        self.attention_mechanism.add_constraint_checker("max_length", self.attention_mechanism._check_max_length)
        st.session_state.attention_mechanism = self.attention_mechanism
        st.session_state.agent = self
        st.session_state.llm = self.llm
        st.session_state.llm_config = self.llm_config

    def create_root_node(self, task_description: str, initial_constraints: Optional[list[str]] = None) -> Node:
        new_node = Node(parent_id=None, task_description=task_description, depth=0)
        st.session_state.node_lookup[new_node.node_id] = new_node
        if initial_constraints:
            for constraint in initial_constraints:
                st.session_state.attention_mechanism.add_constraint(new_node.node_id, constraint)
        st.session_state.attention_mechanism.track_dependencies(None, new_node.node_id)
        return new_node

    def create_child_node(self, parent_node: Node, task_description: str, depth: int) -> Node:
        new_node = Node(parent_id=parent_node.node_id, task_description=task_description, depth=depth)
        st.session_state.node_lookup[new_node.node_id] = new_node
        parent_node.add_child(new_node.node_id)
        st.session_state.attention_mechanism.track_dependencies(parent_node.node_id, new_node.node_id)
        return new_node

    def delete_node_and_children(self, node: Node) -> None:
        for child_id in node.child_ids:
            if child_id in st.session_state.node_lookup:
                self.delete_node_and_children(st.session_state.node_lookup[child_id])
        st.session_state.node_lookup.pop(node.node_id, None)
        st.session_state.attention_mechanism.remove_node(node.node_id)

    def agentFlow(self, action: str, node: Node, regeneration_guidance: str = "") -> None:
        if action == "execute":
            self._execute_node(node)
        elif action == "regenerate":
            self._regenerate_node(node, regeneration_guidance)
        elif action == "delete":
            self.delete_node_and_children(node)
        else:
            raise ValueError(f"Invalid action: {action}")

    def _execute_node(self, node: Node) -> None:
        node.status = STATUS_RUNNING
        prompt = node.build_prompt()
        
        # FIXED: Using st.write and st.code instead of expanders to avoid nesting issues
        st.write("**Executing prompt:**")
        st.code(prompt, language="text")

        for attempt in range(MAX_RETRIES):
            try:
                response = self.llm.generate_content(
                    prompt,
                    generation_config=self.llm_config
                )
                llm_output = response.text
                
                # FIXED: Using st.write and st.code instead of expanders
                st.write(f"**Raw LLM Output (Attempt {attempt + 1}):**")
                st.code(llm_output, language="text")

                node.output = llm_output
                node.process_llm_output(llm_output)

                if not st.session_state.attention_mechanism.check_constraints(node):
                    return
                break

            except Exception as e:
                if handle_node_retryable_error(node, attempt, e):
                    return

        if node.status == STATUS_RUNNING:
            node.status = STATUS_COMPLETED
            if not node.child_ids:
                st.session_state.attention_mechanism.summarize_node(node)

    def _regenerate_node(self, node: Node, regeneration_guidance: str) -> None:
        node.status = STATUS_PENDING
        node.output = ""
        node.error_message = ""
        node.store_in_memory("regeneration_guidance", regeneration_guidance)
        # Remove all child nodes
        for child_id in node.child_ids[:]:
            if child_id in st.session_state.node_lookup:
                self.delete_node_and_children(st.session_state.node_lookup[child_id])
        node.child_ids = []

    def save_session(self, filename: str) -> None:
        data = {
            "node_lookup": {node_id: node.__dict__ for node_id, node in st.session_state.node_lookup.items()},
            "attention_mechanism": {
                "dependency_graph": st.session_state.attention_mechanism.dependency_graph,
                "constraints": st.session_state.attention_mechanism.constraints,
            },
            "root_node_id": st.session_state.root_node_id,
            "global_memory": st.session_state.agent.global_memory.global_context,
            "execution_count": st.session_state.agent.execution_count
        }

        for node_id, node_data in data["node_lookup"].items():
            node_data["local_memory"] = node_data["local_memory"].local_memory

        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

    def load_session(self, filename: str) -> None:
        with open(filename, "r") as f:
            data = json.load(f)

        self.reset_agent()

        st.session_state.root_node_id = data["root_node_id"]
        st.session_state.node_lookup = {}
        
        # Fix the node loading process
        for node_id, node_data in data["node_lookup"].items():
            task_description = node_data.get("task_description")
            if not task_description and "local_memory" in node_data:
                # Try to get task from local memory
                if "task" in node_data["local_memory"]:
                    task_description = node_data["local_memory"]["task"]
                    
            new_node = Node(
                parent_id=node_data["parent_id"], 
                task_description=task_description, 
                depth=node_data.get("depth", 0)
            )
            new_node.node_id = node_data["node_id"]
            new_node.child_ids = node_data["child_ids"]
            new_node.status = node_data["status"]
            new_node.output = node_data["output"]
            new_node.error_message = node_data.get("error_message", "")
            
            # Properly initialize the local memory
            new_node.local_memory = LocalMemory(new_node.node_id)
            if "local_memory" in node_data and isinstance(node_data["local_memory"], dict):
                for key, value in node_data["local_memory"].items():
                    new_node.local_memory.store(key, value)
                    
            st.session_state.node_lookup[node_id] = new_node

        # Set up attention mechanism and constraints
        st.session_state.attention_mechanism.dependency_graph = data["attention_mechanism"]["dependency_graph"]
        st.session_state.attention_mechanism.constraints = data["attention_mechanism"]["constraints"]
        
        # Update global memory
        self.global_memory.update_context(data["global_memory"])
        self.execution_count = data["execution_count"]
