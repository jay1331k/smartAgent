import streamlit as st
import json
import time
import re
from typing import Dict, List, Optional, Any, Callable

from components.utils import parse_constraint, STATUS_FAILED

from components.node import Node


class AttentionMechanism:
    def __init__(self) -> None:
        self.dependency_graph: Dict[str, List[Optional[str]]] = {}
        self.constraints: Dict[str, List[str]] = {}
        self._constraint_checkers = {
            "format": self._check_json_format,
            "contains": self._check_contains_word,
            "max_length": self._check_max_length,
            "code": self._check_has_code,
        }

    def track_dependencies(self, parent_node_id: Optional[str], current_node_id: str) -> None:
        self.add_dependency(current_node_id, parent_node_id)

    def add_dependency(self, dependent_node_id: str, dependency_node_id: Optional[str]) -> None:
        if dependent_node_id not in self.dependency_graph:
            self.dependency_graph[dependent_node_id] = []
        if dependency_node_id and dependency_node_id not in self.dependency_graph[dependent_node_id]:
            self.dependency_graph[dependent_node_id].append(dependency_node_id)

    def get_dependencies(self, node_id: str) -> List[Optional[str]]:
        return self.dependency_graph.get(node_id, [])

    def add_constraint(self, node_id: str, constraint: str) -> None:
        if node_id not in self.constraints:
            self.constraints[node_id] = []
        if constraint not in self.constraints[node_id]:
            self.constraints[node_id].append(constraint)

    def get_constraints(self, node_id: str) -> List[str]:
        return self.constraints.get(node_id, [])

    def update_constraint(self, node_id: str, constraint_index: int, new_constraint: str) -> None:
        if node_id in self.constraints and 0 <= constraint_index < len(self.constraints[node_id]):
            self.constraints[node_id][constraint_index] = new_constraint

    def remove_constraint(self, node_id: str, constraint_index: int) -> None:
        if node_id in self.constraints and 0 <= constraint_index < len(self.constraints[node_id]):
            del self.constraints[node_id][constraint_index]

    def propagate_constraints(self, parent_node_id: str) -> None:
        parent_constraints = self.get_constraints(parent_node_id)
        if parent_node_id in st.session_state.node_lookup:
            for child_id in st.session_state.node_lookup[parent_node_id].child_ids:
                for constraint in parent_constraints:
                    if constraint not in self.get_constraints(child_id):
                        self.add_constraint(child_id, constraint)

    def _summarize_global_context(self) -> None:
        prompt = f"""Summarize the following global context into a concise JSON object with a single field "summary":\n\n{st.session_state.agent.global_memory.get_context()}"""
        try:
            response = st.session_state.agent.llm.generate_content(
                prompt,
                generation_config=st.session_state.llm_config
            )
            response_content = response.text
            if response_content is not None and isinstance(response_content, str):
                summary_json = json.loads(response_content)
                st.session_state.agent.global_memory.update_context(summary_json.get("summary", "Error: Could not summarize global context."))
            else:
                new_context = "Error: LLM returned None or non-string response for global context summarization."
                st.session_state.agent.global_memory.update_context(new_context)
                st.error(new_context)
        except (json.JSONDecodeError, KeyError, Exception) as e:
            new_context = f"Error during summarization: {e}"
            st.session_state.agent.global_memory.update_context(new_context)
            st.error(new_context)

    def summarize_node(self, node: "Node") -> None:
        prompt = f"""Summarize the following task and its result concisely into a JSON object with two fields "task_summary" and "result_summary":

Task: {node.retrieve_from_memory("task")}

Result: {node.output}
"""
        try:
            response = st.session_state.agent.llm.generate_content(
                prompt,
                generation_config=st.session_state.llm_config
            )
            response_content = response.text

            if response_content is not None and isinstance(response_content, str):
                summary_json = json.loads(response_content)
                task_summary = summary_json.get("task_summary", "Task summary not available.")
                result_summary = summary_json.get("result_summary", "Result summary not available.")
                new_context = f"\n- Node {node.node_id} ({node.status}): Task: {task_summary}, Result: {result_summary}"
                st.session_state.agent.global_memory.update_context(st.session_state.agent.global_memory.get_context() + new_context)
            else:
                new_context = f"\n- Node {node.node_id} ({node.status}): Error: LLM returned None or non-string response."
                st.session_state.agent.global_memory.update_context(st.session_state.agent.global_memory.get_context() + new_context)
                st.error(f"Error during summarization of node {node.node_id}: LLM returned None or non-string.")

        except (json.JSONDecodeError, KeyError, Exception) as e:
            new_context = f"\n- Node {node.node_id} ({node.status}): Error during summarization: {e}"
            st.session_state.agent.global_memory.update_context(st.session_state.agent.global_memory.get_context() + new_context)
            st.error(f"Error during summarization of node {node.node_id}: {e}")

        st.session_state.agent.execution_count += 1
        if st.session_state.agent.execution_count % st.session_state.agent.global_context_summary_interval == 0:
            self._summarize_global_context()

    def get_global_context(self) -> str:
        return st.session_state.agent.global_memory.get_context()

    def add_constraint_checker(self, constraint_type, checker) -> None:
        self._constraint_checkers[constraint_type] = checker

    def _check_json_format(self, constraint_value: str, node: "Node") -> bool:
        try:
            json.loads(node.output)
            return True
        except json.JSONDecodeError:
            node.status = STATUS_FAILED
            node.error_message = f"Constraint violated: Output must be in JSON format. Output: {node.output}"
            return False

    def _check_contains_word(self, constraint_value: str, node: "Node") -> bool:
        if constraint_value in node.output:
            return True
        node.status = STATUS_FAILED
        node.error_message = f"Constraint violated: Output must contain '{constraint_value}'. Output: {node.output}"
        return False

    def _check_max_length(self, constraint_value: str, node: "Node") -> bool:
        try:
            max_length = int(constraint_value)
            if len(node.output) <= max_length:
                return True
            node.status = STATUS_FAILED
            node.error_message = f"Constraint violated: Output must be no more than {max_length} characters. Output: {node.output}"
            return False
        except ValueError:
            node.status = STATUS_FAILED
            node.error_message = f"Constraint violated: Invalid max length value '{constraint_value}'"
            return False

    def _check_has_code(self, constraint_value: str, node: "Node") -> bool:
        """Checks if the output has code blocks in the expected format."""
        # Look for code blocks in markdown format: ```language ... ```
        code_blocks = re.findall(r'```(\w*)\n(.*?)\n```', node.output, re.DOTALL)
        
        if not code_blocks:
            node.status = STATUS_FAILED
            node.error_message = f"Constraint violated: Output must contain code blocks. No code blocks found."
            return False
            
        return True

    def check_constraints(self, node: "Node") -> bool:
        for constraint in self.get_constraints(node.node_id):
            parsed_constraint = parse_constraint(constraint)
            checker = self._constraint_checkers.get(parsed_constraint["type"])
            if checker:
                if not checker(parsed_constraint.get("value", ""), node):
                    return False
        return True

    def remove_node(self, node_id: str) -> None:
        self.dependency_graph.pop(node_id, None)
        self.constraints.pop(node_id, None)
        for dependent, sources in self.dependency_graph.items():
            if node_id in sources:
                sources.remove(node_id)
