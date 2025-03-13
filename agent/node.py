# SMARTAGENT/agent/node.py
import streamlit as st
import re
import json
import time
import uuid
import os
import shutil
from typing import Optional, Union
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
        self.local_memory = LocalMemory(self.node_id)  # Node's "notebook"
        self.error_message: str = ""
        self.depth: int = depth

    def store_in_memory(self, key: str, value: str) -> None:
        # Local Memory Usage: Storing data specific to this node.
        self.local_memory.store(key, value)

    def retrieve_from_memory(self, key: str) -> Optional[str]:
        # Local Memory Usage: Retrieving data specific to this node.
        return self.local_memory.retrieve(key)

    def add_child(self, child_node: "Node") -> None:
        self.child_ids.append(child_node.node_id)

    def get_parent_node(self) -> Optional["Node"]:
        return st.session_state.node_lookup.get(self.parent_id)

    def _process_store_command(self, line: str) -> None:
        match = re.match(r"STORE\s+(.+?)\s+(.+)", line)
        if match:
            key, value = match.groups()
            self.store_in_memory(key, value)  # Using Local Memory

    def _process_retrieve_command(self, line: str) -> Optional[str]:
        match = re.match(r"RETRIEVE\s+(.+)", line)
        if match:
            key = match.group(1)
            return self.retrieve_from_memory(key)  # Using Local Memory
        return None

    def _process_query_parent_command(self, line: str) -> Optional[str]:
        match = re.match(r"QUERY_PARENT\s+(.+)", line)
        if match:
            key = match.group(1)
            parent_node = self.get_parent_node()
            # Parent Memory Access: Accessing the parent's Local Memory.
            return parent_node.retrieve_from_memory(key) if parent_node else None
        return None

    def _extract_json(self, text: str) -> Union[dict, None]:
        """
        Extracts the first valid JSON object from a string.
        """
        match = re.search(r"\{[^{}]*(?:(?:\{[^{}]*\})*[^{}]*)*\}", text)
        if match:
            json_str = match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                return None
        return None

    def _process_decompose_command(self, parsed_json: dict) -> bool:
        """Processes the DECOMPOSE command."""
        if "subtasks" in parsed_json:
            sub_tasks_data = parsed_json["subtasks"]

            if not isinstance(sub_tasks_data, list) or not all(isinstance(item, dict) for item in sub_tasks_data):
                self.status = "failed"
                self.error_message = "DECOMPOSE command must contain a JSON array of task objects."
                return False

            if (self.depth + 1) > st.session_state.agent.max_depth:
                self.status = "failed"
                self.error_message = f"Max depth of {st.session_state.agent.max_depth} reached. Cannot Decompose Further."
                return False

            for task_data in sub_tasks_data:
                task_description = task_data.get("task_description")
                if not task_description:
                    self.status = "failed"
                    self.error_message = "Each sub-task must have a 'task_description'."
                    return False
                # Crucially, *no* UUID is passed here.  The child creates its own.
                child_node = st.session_state.agent.create_child_node(self, task_description, self.depth + 1)

                constraints = task_data.get("constraints", [])
                for constraint in constraints:
                    if isinstance(constraint, str):
                        st.session_state.attention_mechanism.add_constraint(child_node.node_id, constraint)

            return True  # Decomposition successful
        return False

    def extract_constraints(self, llm_output: str) -> list[str]:
        """Extracts constraints from the LLM output."""
        constraints = []
        try:
            match = re.search(r"Constraints:\s*```json\s*([\s\S]*?)\s*```", llm_output, re.IGNORECASE)
            if match:
                json_str = match.group(1)
                constraints_data = json.loads(json_str)
                for item in constraints_data:
                    if isinstance(item, dict) and "constraint" in item:
                        constraints.append(item["constraint"])
                    elif isinstance(item, str):
                        constraints.append(item)
        except json.JSONDecodeError:
            st.error("Error decoding JSON in extract_constraints.")
            self.error_message = "Error: Invalid JSON format for constraints."
            self.status = "failed"
        return constraints

    def extract_dependencies(self, llm_output: str) -> list[dict]:
        """Extracts and validates inter-node dependencies."""
        dependencies = []
        try:
            match = re.search(r"Dependencies:\s*```json\s*([\s\S]*?)\s*```", llm_output, re.IGNORECASE)
            if match:
                json_str = match.group(1)
                dependencies_data = json.loads(json_str)
                for dep in dependencies_data:
                    if isinstance(dep, dict) and "dependent_node" in dep and "source_node" in dep:
                        dep_node = dep["dependent_node"]
                        src_node = dep["source_node"]
                        try:
                            uuid.UUID(dep_node)
                            uuid.UUID(src_node)
                            dependencies.append({"dependent_node": dep_node, "source_node": src_node})
                        except ValueError:
                            st.error(f"Invalid UUID format in dependency: {dep_node} or {src_node}")
                            continue
        except json.JSONDecodeError:
            st.error("Error decoding JSON in extract_dependencies.")
            self.error_message = "Error: Invalid JSON format for dependencies."
            self.status = "failed"
        return dependencies
    def _process_file_command(self, llm_output:str) -> None:
        """Processes file-related commands from the LLM output."""

        #Create Directory
        match = re.search(r"CREATE_DIRECTORY\s+(.+)", llm_output)
        if match:
            dir_path = match.group(1).strip()
            self._create_directory(dir_path)
            return

        #Create File
        match = re.search(r"CREATE_FILE\s+(.+)", llm_output)
        if match:
            file_path = match.group(1).strip()
            self._create_file(file_path)
            return

        #Write to File
        match = re.search(r"WRITE_FILE\s+(.+?)\s+```([\s\S]*)```", llm_output, re.DOTALL) #Seperate path and content using ```
        if match:
            file_path = match.group(1).strip()
            content = match.group(2).strip()
            self._write_to_file(file_path, content)
            return

        #Read File
        match = re.search(r"READ_FILE\s+(.+)", llm_output)
        if match:
            file_path = match.group(1).strip()
            content = self._read_file(file_path)
            if content: # Check to prevent storing if _read_file returned ""
                self.output = f"Content of {file_path}:\n```\n{content}\n```" #For display purposes
            return

        #Delete File
        match = re.search(r"DELETE_FILE\s+(.+)", llm_output)
        if match:
            file_path = match.group(1).strip()
            self._delete_file(file_path)
            return

        #Delete Directory
        match = re.search(r"DELETE_DIRECTORY\s+(.+)", llm_output)
        if match:
            dir_path = match.group(1).strip()
            self._delete_directory(dir_path)
            return

    def process_llm_output(self, llm_output: str) -> None:
        """Processes the raw LLM output."""
        self._process_file_command(llm_output)
        parsed_json = self._extract_json(llm_output)

        if parsed_json:
            if "plan" in parsed_json:
                plan = parsed_json["plan"]
                if isinstance(plan, dict):
                    if "sub_tasks" in plan:
                        if self._process_decompose_command({"subtasks": plan["sub_tasks"]}):
                            dependencies = self.extract_dependencies(llm_output)
                            for dep in dependencies:
                                st.session_state.attention_mechanism.add_dependency(dep['dependent_node'], dep['source_node'])
                            return
                    elif "task" in plan:
                        self.output = plan["task"]
                        return
            elif "subtasks" in parsed_json:
                if self._process_decompose_command(parsed_json):
                    dependencies = self.extract_dependencies(llm_output)
                    for dep in dependencies:
                        st.session_state.attention_mechanism.add_dependency(dep['dependent_node'], dep['source_node'])
                    return
            elif "result" in parsed_json:
                self.output = parsed_json["result"]
                return

            self.error_message = "LLM output did not contain expected 'subtasks', 'result', or 'plan' key."
            st.error(self.error_message)
            self.status = "failed"
            return

        else:
            self.error_message = "No valid JSON found in LLM output."
            st.error(self.error_message)
            self.status = "failed"
            return

        self.store_in_memory("raw_llm_output", llm_output)

    def _create_directory(self, dir_path: str) -> None:
        """Creates a directory."""
        try:
            os.makedirs(dir_path, exist_ok=True)
            self.store_in_memory(f"directory_created_{dir_path}", "success")
        except Exception as e:
            self.status = "failed"
            self.error_message = f"Error creating directory '{dir_path}': {e}"
            st.error(self.error_message)

    def _create_file(self, file_path: str) -> None:
        """Creates an empty file."""
        try:
            with open(file_path, "w") as f:
                pass
            self.store_in_memory(f"file_created_{file_path}", "success")
        except Exception as e:
            self.status = "failed"
            self.error_message = f"Error creating file '{file_path}': {e}"
            st.error(self.error_message)

    def _write_to_file(self, file_path: str, content: str) -> None:
        """Writes (or overwrites) content to a file."""
        try:
            with open(file_path, "w") as f:
                f.write(content)
            self.store_in_memory(f"file_written_{file_path}", "success")
        except Exception as e:
            self.status = "failed"
            self.error_message = f"Error writing to file '{file_path}': {e}"
            st.error(self.error_message)

    def _read_file(self, file_path: str) -> str:
        """Reads content from a file."""
        try:
            with open(file_path, "r") as f:
                content = f.read()
            self.store_in_memory(f"file_read_{file_path}", content)
            return content
        except Exception as e:
            self.status = "failed"
            self.error_message = f"Error reading file '{file_path}': {e}"
            st.error(self.error_message)
            return ""

    def _delete_file(self, file_path: str) -> None:
        """Deletes a file."""
        try:
            os.remove(file_path)
            self.store_in_memory(f"file_deleted_{file_path}", "success")
        except Exception as e:
            self.status = "failed"
            self.error_message = f"Error deleting file '{file_path}': {e}"
            st.error(self.error_message)

    def _delete_directory(self, dir_path: str) -> None:
        """Deletes a directory and its contents."""
        try:
            shutil.rmtree(dir_path)
            self.store_in_memory(f"directory_deleted_{dir_path}", "success")
        except Exception as e:
            self.status = "failed"
            self.error_message = f"Error deleting directory '{dir_path}': {e}"
            st.error(self.error_message)
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

        regeneration_guidance = self.retrieve_from_memory("regeneration_guidance")
        guidance_str = f"Additional Human Guidance: {regeneration_guidance}\n" if regeneration_guidance else ""

        prompt = f"""You are a general-purpose assistant.  You can either directly solve the task below, or break it down into smaller sub-tasks.

**Task:** {self.task_description}

{guidance_str}

**Constraints:**
{constraints_str}

{dependencies_str}

**Global Context:**
{global_context}

**Instructions:**

1.  **Consider Direct Solution:** First, try to solve the task directly. If you can provide a complete and satisfactory solution, output a JSON object with a single key: `"result"`.

    ```json
    {{
      "result": "<your solution here>"
    }}
    ```

2.  **Consider Decomposition:** If the task is too complex to solve directly, or if it clearly requires multiple steps, break it down into smaller, manageable sub-tasks.  Output a JSON object with a single key: `"subtasks"`.

    ```json
    {{
      "subtasks": [
        {{
          "task_description": "<detailed description of sub-task 1>",
          "constraints": []
        }},
        {{
          "task_description": "<detailed description of sub-task 2>",
          "constraints": []
        }},
        ...
      ]
    }}
    ```
    *   **Do NOT generate UUIDs for subtasks.** The system will handle those.
    * **Constraints:** Should be simple strings describing the constraint.  Examples: "format: json", "max_length: 100", "contains: keyword".
    *   **Detailed Task Descriptions:** Each sub-task description should be VERY detailed and specific, explaining exactly what needs to be done.

3. **File Operations (Optional):** If the task involves creating, reading, writing, or deleting files, you can use the following commands *outside* the JSON:

    *   `CREATE_DIRECTORY <directory_path>`
    *   `CREATE_FILE <file_path>`
    *   `WRITE_FILE <file_path> \`\`\`<file_content>\`\`\``
    *   `READ_FILE <file_path>`
    *   `DELETE_FILE <file_path>`
    *   `DELETE_DIRECTORY <directory_path>`

4.  **Dependencies (Optional, Separate from JSON):** If there are dependencies BETWEEN sub-tasks, provide them in a SEPARATE JSON block like this, AFTER the main JSON output:
    ```json
    Dependencies:
     ```json
    [
        {{ "dependent_node": "<uuid_of_dependent_node>", "source_node": "<uuid_of_node_it_depends_on>" }}
    ]
    ```

**Output Format:**

*   **Prioritize JSON:**  Always try to return a valid JSON response, either `{{"result": ...}}` or `{{"subtasks": [...]}}`.
*   **File Commands Outside JSON:**  File operation commands should be *outside* the JSON.
* **Dependencies Outside JSON:** If you provide dependency, keep it outside the main JSON.
"""
        return prompt