import streamlit as st
import uuid
import os
import re
import json
from typing import Optional, Dict, Any, List
from components.memory import LocalMemory
from components.utils import extract_json_from_text, extract_code_blocks, STATUS_PENDING, STATUS_RUNNING, STATUS_COMPLETED, STATUS_FAILED

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
        
        # Add depth information for hierarchical awareness
        prompt.append(f"Current depth level: {self.depth}")
        
        # Add special instructions for root nodes to decompose tasks
        if self.depth == 0:
            prompt.append("""
Instructions:
1. Decompose this complex task into manageable subtasks in a hierarchical structure
2. Return your response in JSON format with a 'subtasks' array
3. Each subtask should be a string describing a specific part of the task
4. Aim for 3-7 subtasks that collectively solve the main task
5. Make sure each subtask is clear and focused on a specific aspect
""")
        
        # Add special instructions for code generation if requested
        if self.has_code_constraint():
            prompt.append("""
Code requirements:
1. Include complete, working code in your response
2. Format code blocks with proper markdown: ```language
3. Put file paths in comments at the top of each code block:
   ```python
   # filepath: path/to/file.py
   your code here
   ```
4. Ensure code is ready to be directly saved and executed
5. Include all necessary imports, functions, and components
""")

        # Add constraints
        if 'attention_mechanism' in st.session_state and self.node_id in st.session_state.attention_mechanism.constraints:
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
    
    def has_code_constraint(self) -> bool:
        """Check if this node has a code constraint."""
        if 'attention_mechanism' not in st.session_state:
            return False
        
        constraints = st.session_state.attention_mechanism.get_constraints(self.node_id)
        for constraint in constraints:
            if constraint.startswith('code'):
                return True
        return False
    
    def process_llm_output(self, llm_output: str) -> None:
        """Processes the raw LLM output, extracting subtasks or results."""
        try:
            # First try to extract code files
            code_files = self.extract_code_files()
            if code_files:
                self.store_in_memory("code_files", code_files)
            
            # Try to parse JSON from the output
            parsed_output = extract_json_from_text(llm_output)
            
            if parsed_output:
                # Handle subtasks in the output
                if "subtasks" in parsed_output and isinstance(parsed_output["subtasks"], list):
                    for subtask in parsed_output["subtasks"]:
                        if isinstance(subtask, str):
                            # Simple string subtask
                            self._create_child_node(subtask)
                        elif isinstance(subtask, dict) and "task_description" in subtask:
                            # Subtask with description
                            self._create_child_node(subtask["task_description"])
                    
                    # Check for code in the JSON
                    self._extract_code_from_json(parsed_output)
                    
                # Handle result in the output
                elif "result" in parsed_output:
                    self.store_in_memory("result", parsed_output["result"])
                    # Check for code in the result
                    self._extract_code_from_json(parsed_output)
                # Handle other parsed output
                else:
                    # Store the entire parsed output if no specific keys are found
                    self.store_in_memory("result", parsed_output)
                    # Check for code in the JSON
                    self._extract_code_from_json(parsed_output)
            else:
                # If no JSON found, check if we already found code blocks
                if not code_files:
                    # If no JSON or code blocks, store the entire output as the result
                    self.store_in_memory("result", llm_output)

        except Exception as e:
            self.status = STATUS_FAILED
            self.error_message = f"Error processing LLM output: {str(e)}"
        
    def _extract_code_from_json(self, json_data: Dict[str, Any]) -> None:
        """Extract code from JSON fields and store as code files."""
        if not json_data:
            return
            
        code_files = {}
        
        # Look for code fields in any part of the JSON structure
        def search_for_code(obj, prefix=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    key_lower = key.lower()
                    # Check if this is a code field
                    if isinstance(value, str) and ('code' in key_lower or 'implementation' in key_lower or key_lower.endswith('_py') or key_lower.endswith('_js')):
                        # Generate a filename based on the key
                        ext = 'py' if 'python' in key_lower or key_lower.endswith('_py') else 'js'
                        if 'html' in key_lower:
                            ext = 'html'
                        elif 'css' in key_lower:
                            ext = 'css'
                        filename = f"{key.lower().replace('_code', '').replace('code_', '')}.{ext}"
                        if prefix:
                            filename = f"{prefix}_{filename}"
                        code_files[filename] = value
                    elif isinstance(value, (dict, list)):
                        new_prefix = f"{prefix}_{key}" if prefix else key
                        search_for_code(value, new_prefix)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    search_for_code(item, f"{prefix}_{i}" if prefix else str(i))
        
        # Start recursive search
        search_for_code(json_data)
        
        # Store found code files
        if code_files:
            current_files = self.retrieve_from_memory("code_files") or {}
            current_files.update(code_files)
            self.store_in_memory("code_files", current_files)

    def extract_code_files(self) -> Dict[str, str]:
        """Extract code files from the output if present."""
        result = {}
        
        # Pattern 1: Properly formatted code blocks with filepath comments
        pattern1 = r'```(?:python|javascript|html|css)\n# filepath: (.*?)\n(.*?)```'
        matches1 = re.findall(pattern1, self.output, re.DOTALL)
        for filepath, content in matches1:
            result[filepath] = content.strip()
        
        # Pattern 2: Code blocks without filepath comments
        pattern2 = r'```(?:python|javascript|html|css)?\n(.*?)```'
        matches2 = re.findall(pattern2, self.output, re.DOTALL)
        for i, content in enumerate(matches2):
            result[f"code_block_{i}.py"] = content.strip()
        
        # Also check for any code in memory (e.g., extracted from JSON)
        memory_code_files = self.retrieve_from_memory("code_files")
        if memory_code_files and isinstance(memory_code_files, dict):
            result.update(memory_code_files)
        
        # Log finding to help with debugging
        if result:
            print(f"Found {len(result)} code files: {list(result.keys())}")
        
        return result

    def save_state(self, directory: str = "node_memory") -> None:
        """Save the node's state and memory to disk."""
        self.local_memory.save_to_disk(directory)

    def load_state(self, directory: str = "node_memory") -> bool:
        """Load the node's state and memory from disk."""
        return self.local_memory.load_from_disk(directory)

    def _create_child_node(self, task_description: str) -> None:
        """Creates a child node and adds it to the tree.
        
        Note: This is an internal method that delegates to the Agent method.
        """
        if 'agent' not in st.session_state:
            self.error_message = "Agent not available to create child node"
            return
        
        # Create child node through the agent to ensure proper tracking
        st.session_state.agent.create_child_node(
            parent_node=self, 
            task_description=task_description, 
            depth=self.depth + 1
        )
