import streamlit as st
import os
import sys
import subprocess
import tempfile
from pathlib import Path
import io
import zipfile
import re
import json
import uuid
import time  # Add explicit import for time
from typing import Optional, Dict, List, Any, Union, Tuple  # Add Any to imports

# --- Imports for AI (Gemini) ---
import google.generativeai as genai

# --- Constants (from agent/constants.py) ---
MAX_RETRIES = 3
RETRY_DELAY = 2
MAX_DEPTH = 5  # Example value, adjust as needed
GLOBAL_CONTEXT_SUMMARY_INTERVAL = 5
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

# --- Utility Functions (from agent/utils.py and memory_utils.py) ---
def handle_retryable_error(func, *args, **kwargs):
    retries = 0
    while retries < MAX_RETRIES:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            retries += 1
            if retries >= MAX_RETRIES:
                raise
            print(f"Error: {e}. Retrying in {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)
    raise RuntimeError("Unexpected error in handle_retryable_error")

def parse_constraint(constraint_str):
    result = {"type": "generic", "value": constraint_str}
    format_match = re.match(r"format:\s*(\w+)", constraint_str, re.IGNORECASE)
    if format_match:
        return {"type": "format", "format": format_match.group(1).lower()}
    length_match = re.match(r"max_length:\s*(\d+)", constraint_str, re.IGNORECASE)
    if length_match:
        return {"type": "max_length", "length": int(length_match.group(1))}
    contains_match = re.match(r"contains:\s*(.+)", constraint_str, re.IGNORECASE)
    if contains_match:
        return {"type": "contains", "value": contains_match.group(1).strip()}
    return result

def extract_json_from_text(text):
    json_pattern = r"\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{(?:[^{}])*\}))*\}))*\}"
    match = re.search(json_pattern, text)
    if match:
        json_str = match.group(0)
        try: return json.loads(json_str)
        except json.JSONDecodeError: pass
    code_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    matches = re.findall(code_block_pattern, text)
    for potential_json in matches:
        try: return json.loads(potential_json)
        except json.JSONDecodeError: continue
    result_pattern = r"\{\s*\"result\"[\s\S]*?\}"
    match = re.search(result_pattern, text)
    if match:
        try: return json.loads(match.group(0))
        except json.JSONDecodeError: pass
    subtasks_pattern = r"\{\s*\"subtasks\"[\s\S]*?\}"
    match = re.search(subtasks_pattern, text)
    if match:
        try: return json.loads(match.group(0))
        except json.JSONDecodeError: pass
    return None

def handle_node_retryable_error(node, attempt, exception):
    if attempt < MAX_RETRIES - 1:
        time.sleep(RETRY_DELAY)
        return False
    else:
        node.status = "failed"
        node.error_message = f"Error after {MAX_RETRIES} attempts: {str(exception)}"
        return True

def parse_response(text: str):
    try:
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx >= 0 and end_idx > start_idx:
            json_str = text[start_idx:end_idx+1]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    return text

def safe_serialize(obj):
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [safe_serialize(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): safe_serialize(v) for k, v in obj.items()}
    else:
        return str(obj)

def create_structured_memory(raw_response):
    parsed = parse_response(raw_response)
    memory = {
        "raw_llm_response": raw_response,
        "parsed_response": parsed if isinstance(parsed, dict) else {},
        "is_structured": isinstance(parsed, dict)
    }
    return memory

# --- Local Memory (from agent/memory.py) ---
class LocalMemory:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.local_memory: Dict[str, Any] = {}
    def store(self, key: str, value: Any) -> None:
        self.local_memory[key] = value
    def retrieve(self, key: str) -> Optional[Any]:
        return self.local_memory.get(key)
    def clear(self) -> None:
        self.local_memory = {}
    def get_all(self) -> Dict[str, Any]:
        return self.local_memory
    def save_to_disk(self, directory: str = "node_memory") -> None:
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, f"{self.node_id}.json")
        with open(filepath, 'w') as f:
            json.dump(self.local_memory, f, indent=2)
    def load_from_disk(self, directory: str = "node_memory") -> bool:
        filepath = os.path.join(directory, f"{self.node_id}.json")
        if not os.path.exists(filepath):
            return False
        try:
            with open(filepath, 'r') as f:
                self.local_memory = json.load(f)
            return True
        except Exception:
            return False

# --- Global Memory (from agent/memory.py) ---
class GlobalMemory:
    def __init__(self):
        self.global_context: str = "This agent can solve tasks by breaking them down into subtasks."
        self.shared_data: Dict[str, Any] = {}
    def update_context(self, context: str) -> None:
        self.global_context = context
    def get_context(self) -> str:
        return self.global_context
    def store(self, key: str, value: Any) -> None:
        self.shared_data[key] = value
    def retrieve(self, key: str) -> Optional[Any]:
        return self.shared_data.get(key)
    def save_to_disk(self, filepath: str = "global_memory.json") -> None:
        with open(filepath, 'w') as f:
            json.dump({
                "global_context": self.global_context,
                "shared_data": self.shared_data
            }, f, indent=2)
    def load_from_disk(self, filepath: str = "global_memory.json") -> bool:
        if not os.path.exists(filepath):
            return False
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.global_context = data.get("global_context", self.global_context)
                self.shared_data = data.get("shared_data", {})
            return True
        except Exception:
            return False

# --- Node Class (from agent/node.py) ---
class Node:
    def __init__(self, parent_id: Optional[str] = None, task_description: Optional[str] = None, depth: int = 0):
        self.node_id = str(uuid.uuid4())
        self.depth = depth
        self.local_memory = LocalMemory(self.node_id)
        self.parent_id = parent_id
        self.child_ids: List[str] = []
        self.status = STATUS_PENDING
        self.output = ""  # Store raw output
        self.error_message = ""

        if task_description:
            self.store_in_memory("task", task_description)

    def store_in_memory(self, key: str, value: Any) -> None:
        self.local_memory.store(key, value)

    def retrieve_from_memory(self, key: str) -> Any:
        return self.local_memory.retrieve(key)

    def add_child(self, child_id: str) -> None:
        if child_id not in self.child_ids:
            self.child_ids.append(child_id)

    def set_parent(self, parent_id: str) -> None:
        self.parent_id = parent_id

    def update_status(self, status: str) -> None:
        self.status = status
        self.store_in_memory("status", status)

    def get_summary(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "status": self.status,
            "parent_id": self.parent_id,
            "children": self.child_ids,
            "task": self.retrieve_from_memory("task") or "",
            "result": self.retrieve_from_memory("result") or ""
        }

    def save_state(self, directory: str = "node_memory") -> None:
        self.local_memory.save_to_disk(directory)

    def load_state(self, directory: str = "node_memory") -> bool:
        return self.local_memory.load_from_disk(directory)

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
        new_node = Node(parent_id=self.node_id, task_description=task_description, depth=depth)
        st.session_state.node_lookup[new_node.node_id] = new_node
        self.add_child(new_node.node_id)
        st.session_state.attention_mechanism.track_dependencies(self.node_id, new_node.node_id)
        return new_node

# --- Attention Mechanism (from agent/attention_mechanism.py) ---
class AttentionMechanism:
    def __init__(self) -> None:
        self.dependency_graph: Dict[str, List[Optional[str]]] = {}
        self.constraints: Dict[str, List[str]] = {}
        self._constraint_checkers = {
            "format": self._check_json_format,
            "contains": self._check_contains_word,
            "max_length": self._check_max_length,
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

Task: {node.task_description}

Result: {node.output}
"""
        try:
            response = st.session_state.agent.llm.generate_content(
                prompt,
                generation_config= st.session_state.llm_config
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
            node.status = "failed"
            node.error_message = f"Constraint violated: Output must be in JSON format. Output: {node.output}"
            return False

    def _check_contains_word(self, constraint_value: str, node: "Node") -> bool:
        if constraint_value in node.output:
            return True
        node.status = "failed"
        node.error_message = f"Constraint violated: Output must contain '{constraint_value}'. Output: {node.output}"
        return False

    def _check_max_length(self, constraint_value: str, node: "Node") -> bool:
        try:
            max_length = int(constraint_value)
            if len(node.output) <= max_length:
                return True
            node.status = "failed"
            node.error_message = f"Constraint violated: Output must be no more than {max_length} characters. Output: {node.output}"
            return False
        except ValueError:
            node.status = "failed"
            node.error_message = f"Constraint violated: Invalid max length value '{constraint_value}'"
            return False

    def check_constraints(self, node: "Node") -> bool:
        for constraint in self.get_constraints(node.node_id):
            constraint_type, constraint_value = parse_constraint(constraint)
            checker = self._constraint_checkers.get(constraint_type)
            if checker:
                if not checker(constraint_value, node):
                    return False
        return True

    def remove_node(self, node_id: str) -> None:
        self.dependency_graph.pop(node_id, None)
        self.constraints.pop(node_id, None)
        for dependent, sources in self.dependency_graph.items():
            if node_id in sources:
                sources.remove(node_id)

# --- Agent Class (from agent/agent.py) ---
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
        st.session_state.clear()
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
        node.status = "running"
        prompt = node.build_prompt()
        st.write(f"Prompt: {prompt}")

        for attempt in range(MAX_RETRIES):
            try:
                response = self.llm.generate_content(
                    prompt,
                    generation_config=st.session_state.llm_config
                )
                llm_output = response.text
                st.write(f"Raw LLM Output (Attempt {attempt + 1}):")
                st.code(llm_output, language="text")

                node.output = llm_output
                node.process_llm_output(llm_output)

                if not st.session_state.attention_mechanism.check_constraints(node):
                    return
                break

            except Exception as e:
                if handle_node_retryable_error(node, attempt, e):
                    return

        if node.status == "running":
            node.status = "completed"
            if not node.child_ids:
                st.session_state.attention_mechanism.summarize_node(node)

    def _regenerate_node(self, node: Node, regeneration_guidance: str) -> None:
        node.status = "pending"
        node.output = ""
        node.error_message = ""
        if not isinstance(node.local_memory, LocalMemory):
            node.local_memory = LocalMemory(node.node_id)
        node.store_in_memory("regeneration_guidance", regeneration_guidance)

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
        for node_id, node_data in data["node_lookup"].items():
            new_node = Node(node_data["parent_id"], node_data["task_description"], node_data["depth"])
            new_node.node_id = node_data["node_id"]
            new_node.child_ids = node_data["child_ids"]
            new_node.status = node_data["status"]
            new_node.output = node_data["output"]
            new_node.error_message = node_data["error_message"]
            new_node.local_memory = LocalMemory(new_node.node_id)
            new_node.local_memory.local_memory = node_data["local_memory"]
            st.session_state.node_lookup[node_id] = new_node

        st.session_state.attention_mechanism.dependency_graph = data["attention_mechanism"]["dependency_graph"]
        st.session_state.attention_mechanism.constraints = data["attention_mechanism"]["constraints"]
        st.session_state.agent.global_memory.update_context(data["global_memory"])
        st.session_state.agent.execution_count = data["execution_count"]

# --- File Explorer (from streamlit_app/file_explorer.py) ---
class FileExplorer:
    def __init__(self, root_path):
        self.root_path = root_path

    def get_file_tree(self):
        file_tree = []
        for root, dirs, files in os.walk(self.root_path):
            rel_path = os.path.relpath(root, self.root_path)
            if rel_path == ".":
                for dir_name in sorted(dirs):
                    file_tree.append({
                        "name": dir_name,
                        "type": "directory",
                        "path": os.path.join(self.root_path, dir_name),
                        "children": []
                    })
                for file_name in sorted(files):
                    file_tree.append({
                        "name": file_name,
                        "type": "file",
                        "path": os.path.join(self.root_path, file_name)
                    })
        return file_tree

    def display_file_tree(self):
        file_tree = self.get_file_tree()
        return self._render_file_tree(file_tree)
    
    def _render_file_tree(self, items):
        for item in items:
            if item["type"] == "directory":
                expander = st.expander(f"📁 {item['name']}")
                with expander:
                    if "children" in item and item["children"]:
                        self._render_file_tree(item["children"])
                    else:
                        # Load directory contents when expanded
                        children = []
                        dir_path = item["path"]
                        try:
                            for entry in os.scandir(dir_path):
                                if entry.is_dir():
                                    children.append({
                                        "name": entry.name,
                                        "type": "directory",
                                        "path": entry.path,
                                        "children": []
                                    })
                                else:
                                    children.append({
                                        "name": entry.name,
                                        "type": "file",
                                        "path": entry.path
                                    })
                            if children:
                                self._render_file_tree(sorted(children, key=lambda x: (x["type"] != "directory", x["name"])))
                            else:
                                st.write("(Empty directory)")
                        except PermissionError:
                            st.write("(Permission denied)")
                        except Exception as e:
                            st.write(f"(Error: {str(e)})")
            else:  # File
                if st.button(f"📄 {item['name']}", key=f"file_{item['path']}"):
                    self.display_file_content(item["path"])
    
    def display_file_content(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                content = file.read()
                st.session_state.current_file = file_path
                st.session_state.file_content = content
        except Exception as e:
            st.error(f"Error opening file: {str(e)}")

# --- Terminal (from streamlit_app/terminal.py) ---
class Terminal:
    def __init__(self, working_directory=None):
        self.working_directory = working_directory or os.getcwd()
        self.history = []
        
    def run_command(self, command):
        if not command.strip():
            return "No command provided."
        
        try:
            # Create a temporary file to capture output
            with tempfile.TemporaryFile(mode='w+t') as stdout_file, tempfile.TemporaryFile(mode='w+t') as stderr_file:
                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=self.working_directory
                )
                
                stdout, stderr = process.communicate(timeout=15)  # 15 second timeout
                result = f"$ {command}\n"
                if stdout:
                    result += stdout
                if stderr:
                    result += f"\nError:\n{stderr}"
                
                self.history.append({
                    "command": command,
                    "result": result,
                    "exit_code": process.returncode
                })
                return result
                
        except subprocess.TimeoutExpired:
            process.kill()
            return f"Command timed out: {command}"
        except Exception as e:
            return f"Error executing command: {str(e)}"
    
    def get_history(self):
        return self.history

# --- Command Line Interface (from streamlit_app/cline_interface.py) ---
class ClineInterface:
    def __init__(self, terminal=None):
        self.terminal = terminal or Terminal()
        
    def display(self):
        st.subheader("Terminal")
        
        # Command input
        command = st.text_input("Enter command:", key="terminal_command")
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("Run"):
                if command:
                    output = self.terminal.run_command(command)
                    if "terminal_history" not in st.session_state:
                        st.session_state.terminal_history = []
                    st.session_state.terminal_history.append(output)
                    st.session_state.terminal_command = ""  # Clear input
        with col2:
            if st.button("Clear History"):
                st.session_state.terminal_history = []
        
        # Display history
        if "terminal_history" in st.session_state and st.session_state.terminal_history:
            for i, entry in enumerate(st.session_state.terminal_history):
                with st.expander(f"Command #{i+1}", expanded=(i == len(st.session_state.terminal_history)-1)):
                    st.code(entry, language="bash")

# --- Streamlit App Interface ---
def initialize_gemini_api():
    api_key = st.secrets.get("GOOGLE_API_KEY", None)
    if api_key:
        genai.configure(api_key=api_key)
        return True
    else:
        st.error("Google API Key not found in secrets. Please configure it.")
        api_key = st.text_input("Enter Google API Key:", type="password")
        if api_key:
            genai.configure(api_key=api_key)
            return True
    return False

def get_model():
    models = [m for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    if not models:
        st.error("No suitable models found. Check your API key.")
        return None
    
    selected_model = st.session_state.get('selected_model', "gemini-2.0-pro-exp-02-05")
    model = genai.GenerativeModel(selected_model)
    return model

def initialize_agent():
    if 'agent' not in st.session_state:
        llm = get_model()
        if not llm:
            return False
        
        # Default LLM configuration
        llm_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        agent = Agent(
            llm=llm,
            llm_config=llm_config,
            global_context="This agent can break down complex tasks into manageable subtasks."
        )
        st.session_state.agent = agent
        agent.setup_agent()
    return True

def render_node_tree(node_id, depth=0):
    if not node_id or node_id not in st.session_state.node_lookup:
        return

    node = st.session_state.node_lookup[node_id]
    indent = " " * (depth * 4)
    
    # Create an expander for this node
    task = node.retrieve_from_memory("task") or "Task not specified"
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
            
        # Show node output/result
        if node.output:
            with st.expander("Raw Output"):
                st.code(node.output)
                
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

        # Render child nodes
        if node.child_ids:
            st.write("**Subtasks:**")
            for child_id in node.child_ids:
                render_node_tree(child_id, depth + 1)
        elif node.status == STATUS_COMPLETED:
            st.write("No subtasks created.")

def main():
    st.set_page_config(page_title="Smart Agent Interface", layout="wide", page_icon="🤖")
    
    st.title("🤖 Smart Agent Interface")
    
    # Initialize Gemini API
    if not initialize_gemini_api():
        st.stop()
    
    # Initialize Agent
    if not initialize_agent():
        st.stop()
    
    # Main app tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Agent", "Files", "Terminal", "Settings"])
    
    with tab1:
        # Agent Interface
        st.header("Smart Agent")
        
        # Task input and running
        if 'root_node_id' not in st.session_state:
            with st.form("task_form"):
                task_description = st.text_area("Enter Task Description:", height=100)
                initial_constraint = st.text_input("Initial Constraint (optional):")
                submitted = st.form_submit_button("Run Task")
                
                if submitted and task_description:
                    initial_constraints = [initial_constraint] if initial_constraint else None
                    st.session_state.agent.run(task_description, initial_constraints)
                    st.experimental_rerun()
        
        # Display task tree
        else:
            st.subheader("Task Tree")
            render_node_tree(st.session_state.root_node_id)
            
            # New task button
            if st.button("Start New Task"):
                st.session_state.agent.reset_agent()
                st.experimental_rerun()
            
            # Save/Load session
            col1, col2 = st.columns(2)
            with col1:
                save_filename = st.text_input("Save session as:", value="agent_session.json")
                if st.button("Save Session"):
                    st.session_state.agent.save_session(save_filename)
                    st.success(f"Session saved to {save_filename}")
            
            with col2:
                uploaded_file = st.file_uploader("Load session from file:", type="json")
                if uploaded_file and st.button("Load Session"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    
                    try:
                        st.session_state.agent.load_session(tmp_path)
                        st.success("Session loaded successfully")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error loading session: {str(e)}")
                    finally:
                        os.unlink(tmp_path)
    
    with tab2:
        # File Explorer
        st.header("File Explorer")
        
        # Directory input
        default_dir = st.session_state.get("explorer_dir", os.getcwd())
        dir_path = st.text_input("Directory Path:", value=default_dir)
        if st.button("Browse"):
            st.session_state.explorer_dir = dir_path
            st.experimental_rerun()
        
        # Show file explorer
        if os.path.isdir(dir_path):
            explorer = FileExplorer(dir_path)
            explorer.display_file_tree()
            
            # Display selected file
            if 'current_file' in st.session_state and 'file_content' in st.session_state:
                st.subheader(f"File: {os.path.basename(st.session_state.current_file)}")
                st.code(st.session_state.file_content, language="python")
        else:
            st.error("Invalid directory path")

        # --- File Explorer Tab ---
        st.write("## File Explorer")
        st.write("Browse and manage files in your project")

    
    with tab3:
        # Terminal
        if 'terminal' not in st.session_state:
            st.session_state.terminal = Terminal()
        
        cli = ClineInterface(st.session_state.terminal)
        cli.display()
    
    with tab4:
        # Settings
        st.header("Settings")
        
        # Model selection
        st.subheader("Model Settings")
        models = [
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-2.0-pro-exp-02-05",
            "gemini-2.0-flash-thinking-exp-01-21",
            "gemini-2.0-flash-exp"
        ]
        selected_model = st.selectbox("Select Model", models, index=models.index("gemini-2.0-pro-exp-02-05") if "gemini-2.0-pro-exp-02-05" in models else 0)
        if selected_model != st.session_state.get('selected_model'):
            st.session_state.selected_model = selected_model

        # LLM configuration
        st.subheader("LLM Configuration")
        temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.llm_config["temperature"], 0.1)
        top_p = st.slider("Top P", 0.0, 1.0, st.session_state.llm_config["top_p"], 0.05)
        top_k = st.slider("Top K", 1, 100, st.session_state.llm_config["top_k"], 1)
        max_tokens = st.slider("Max Output Tokens", 100, 8192, st.session_state.llm_config["max_output_tokens"], 100)
        
        if st.button("Apply Settings"):
            st.session_state.llm_config = {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_output_tokens": max_tokens,
            }
            if 'agent' in st.session_state:
                st.session_state.agent.llm = get_model()
                st.session_state.agent.llm_config = st.session_state.llm_config
            st.success("Settings applied successfully")
            
        # Agent configuration
        st.subheader("Agent Configuration")
        max_depth = st.slider("Max Tree Depth", 1, 10, st.session_state.agent.max_depth, 1)
        summary_interval = st.slider("Global Context Summary Interval", 1, 20, st.session_state.agent.global_context_summary_interval, 1)
        
        if st.button("Apply Agent Settings"):
            st.session_state.agent.max_depth = max_depth
            st.session_state.agent.global_context_summary_interval = summary_interval
            st.success("Agent settings applied successfully")

# Add missing import
import time

# Run the app
if __name__ == "__main__":
    main()

# Add a footer with app information
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    <p>AI-Powered IDE with Hierarchical Task Decomposition</p>
    <p>Built with Streamlit and Google Gemini</p>
</div>
""", unsafe_allow_html=True)
