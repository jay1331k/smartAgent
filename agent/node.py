# SMARTAGENT/agent/node.py (Continued)
import streamlit as st
import re
import json
import time
import uuid
from typing import Optional
from .memory import LocalMemory
from .utils import handle_retryable_error

class Node:
    def __init__(self, parent_id: Optional[str], task_description: str, depth: int) -> None:
        self.node_id: str = str(uuid.uuid4())
        self.parent_id: Optional[str] = parent_id
        self.child_ids: list[str] = []
        self.task_description: str = task_description
        self.status: str = "pending"  # pending, running, completed, failed, overridden
        self.output: str = ""
        self.local_memory = LocalMemory(self.node_id)  # Use LocalMemory class
        self.error_message: str = ""
        self.depth: int = depth

    def store_in_memory(self, key: str, value: str) -> None:
        """Stores a key-value pair in the node's local memory."""
        self.local_memory.store(key, value)

    def retrieve_from_memory(self, key: str) -> Optional[str]:
        """Retrieves a value from the node's local memory by key."""
        return self.local_memory.retrieve(key)

    def add_child(self, child_node: "Node") -> None:
        """Adds a child node ID to the list of children."""
        self.child_ids.append(child_node.node_id)

    def get_parent_node(self) -> Optional["Node"]:
        """Retrieves the parent node object from the session state."""
        return st.session_state.node_lookup.get(self.parent_id)

    def _process_store_command(self, line: str) -> None:
        """Processes a STORE command from the LLM output."""
        match = re.match(r"STORE\s+(.+?)\s+(.+)", line)
        if match:
            key, value = match.groups()
            self.store_in_memory(key, value)

    def _process_retrieve_command(self, line: str) -> Optional[str]:
        """Processes a RETRIEVE command from the LLM output."""
        match = re.match(r"RETRIEVE\s+(.+)", line)
        if match:
            key = match.group(1)
            return self.retrieve_from_memory(key)
        return None

    def _process_query_parent_command(self, line: str) -> Optional[str]:
        """Processes a QUERY_PARENT command."""
        match = re.match(r"QUERY_PARENT\s+(.+)", line)
        if match:
            key = match.group(1)
            parent_node = self.get_parent_node()
            return parent_node.retrieve_from_memory(key) if parent_node else None
        return None

    def _process_decompose_command(self, line: str, llm_output: str) -> bool:
        """Processes a DECOMPOSE command, creating child nodes."""
        match = re.match(r"DECOMPOSE\s+(.+)", line)
        if match:
            sub_tasks_str = match.group(1)
            sub_tasks = [t.strip() for t in sub_tasks_str.split(';') if t.strip()]
            if (self.depth + 1) > st.session_state.agent.max_depth:
                self.status = "failed"
                self.error_message = f"Max depth of {st.session_state.agent.max_depth} reached. Cannot Decompose Further."
                return False
            if not sub_tasks:
                self.status = "failed"
                self.error_message = "DECOMPOSE command used but no subtasks provided."
                return False

            constraints = self.extract_constraints(llm_output)
            for task_description in sub_tasks:
                child_node = st.session_state.agent.create_child_node(self, task_description, self.depth + 1)
                for constraint in constraints:
                    st.session_state.attention_mechanism.add_constraint(child_node.node_id, constraint)

            st.session_state.attention_mechanism.propagate_constraints(self.node_id)
            return True  # Decomposition successful
        return False  # No decomposition command found

    def extract_constraints(self, llm_output: str) -> list[str]:
        """Extracts constraints from the LLM output (if present)."""
        constraints = []
        try:
            match = re.search(r"Constraints:\s*```json\s*([\s\S]*?)\s*```", llm_output, re.IGNORECASE)
            if match:
                json_str = match.group(1)
                constraints_data = json.loads(json_str)
                for item in constraints_data:
                    if isinstance(item, dict) and "constraint" in item:
                        constraints.append(item["constraint"])
        except json.JSONDecodeError:
            st.error("Error decoding JSON in extract_constraints.  Check LLM output.")
            self.error_message = "Error: Invalid JSON format for constraints."
            self.status = "failed"

        return constraints

    def _extract_inter_node_dependencies(self, llm_output: str) -> None:
        """
        Extracts and stores inter-node dependencies from the LLM output.
        """
        try:
            match = re.search(r"Dependencies:\s*```json\s*([\s\S]*?)\s*```", llm_output, re.IGNORECASE)
            if match:
                json_str = match.group(1)
                dependencies_data = json.loads(json_str)
                for dep in dependencies_data:
                    if isinstance(dep, dict) and "dependent_node" in dep and "source_node" in dep:
                        dependent_node_str = dep["dependent_node"]
                        source_node_str = dep["source_node"]

                        try:
                            dependent_node = str(uuid.UUID(dependent_node_str))
                            source_node = str(uuid.UUID(source_node_str))
                        except ValueError:
                            st.error(f"Invalid UUID format in dependencies: {dependent_node_str} or {source_node_str}")
                            continue

                        st.session_state.attention_mechanism.add_dependency(dependent_node, source_node)
                        self.store_in_memory(f"dependency_{dependent_node}", source_node)

        except json.JSONDecodeError:
            st.error("Error decoding JSON in _extract_inter_node_dependencies. Check LLM output.")
            self.error_message = "Error: Invalid JSON format for dependencies."
            self.status = "failed"
        except Exception as e:
            st.error(f"Error in _extract_inter_node_dependencies: {e}")
            self.error_message = f"Error processing dependencies: {e}"
            self.status = "failed"

    def process_llm_output(self, llm_output: str) -> None:
        """Processes the raw LLM output."""
        lines = llm_output.splitlines()
        for line in lines:
            line = line.strip()
            if line.startswith("STORE"):
                self._process_store_command(line)
            elif line.startswith("RETRIEVE"):
                retrieved_value = self._process_retrieve_command(line)
            elif line.startswith("QUERY_PARENT"):
                parent_value = self._process_query_parent_command(line)
            elif line.startswith("DECOMPOSE"):
                if self._process_decompose_command(line, llm_output):
                    return

        self._extract_inter_node_dependencies(llm_output)
        self.store_in_memory("raw_llm_output", llm_output)

    def build_prompt(self) -> str:
        """Constructs the prompt for the LLM."""
        constraints = st.session_state.attention_mechanism.get_constraints(self.node_id)
        constraints_str = "\n".join([f"- {c}" for c in constraints]) if constraints else "None"
        global_context = st.session_state.attention_mechanism.get_global_context()

        dependencies = st.session_state.attention_mechanism.get_dependencies(self.node_id)
        dependencies_str = ""
        if dependencies:
            dependencies_str = "\nInter-Node Dependencies:\n"
            for dep_node_id in dependencies:
                dep_node = st.session_state.node_lookup.get(dep_node_id)
                if dep_node:
                    dependencies_str += f"- Requires output from Node {dep_node_id} (Task: {dep_node.task_description})\n"
                else:
                    dependencies_str += f"- Requires output from Node {dep_node_id} (Node not found - check dependency graph!)\n"
        # Regeneration Guidance
        regeneration_guidance = self.retrieve_from_memory("regeneration_guidance")
        guidance_str = f"Additional Human Guidance: {regeneration_guidance}\n" if regeneration_guidance else ""

        prompt = f"""You are a helpful assistant tasked with solving the following:

Task: {self.task_description}

{guidance_str}

Constraints:
{constraints_str}

{dependencies_str}

Global Context:
{global_context}

You have access to local memory.  Use the following commands to interact with it:
- STORE <key> <value>: Store information in memory.  Example: STORE tools search_engine
- RETRIEVE <key>: Retrieve information from memory. Example: RETRIEVE tools
- QUERY_PARENT <key>: Retrieve information from the parent's memory. Example: QUERY_PARENT overall_goal

If the task is even slightly big for one llm call to solve directly, you can decompose it into smaller sub-tasks:
- DECOMPOSE <task_1>; <task_2>; ... ; <task_n>: Break down the task into sub-tasks.  Example: DECOMPOSE Find contact info; Email the contact; Log the email
  You may also provide CONSTRAINTS for the child nodes in JSON Format.
  Constraints:
    ```json
    [
        {{"constraint": "format: json"}},
        {{"constraint": "max_length: 200"}}
    ]
    ```

  You may also provide inter-node dependencies.  *Crucially*, these dependencies must use the UUIDs of the nodes. Example:
  Dependencies:
    ```json
        [
            {{"dependent_node": "123e4567-e89b-12d3-a456-426614174000", "source_node": "123e4567-e89b-12d3-a456-426614174001"}}
        ]
    ```

Previous Memory Operations:
{self.local_memory.retrieve("memory_operations_history", "")}

Provide your solution. If you can solve the task directly, provide the solution in JSON format. If not, use DECOMPOSE. The final output MUST be valid JSON.
"""
        return prompt