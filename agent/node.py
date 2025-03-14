# SMARTAGENT/agent/node.py
import streamlit as st
import re
import json
import time
import uuid
import os
import shutil
import subprocess
import requests
from urllib.parse import urlparse
from typing import Optional, Union, Dict, Any, List
from .memory import LocalMemory
from .utils import handle_retryable_error, extract_json_from_text
from .constants import STATUS_PENDING, STATUS_RUNNING, STATUS_COMPLETED, STATUS_FAILED, STATUS_OVERRIDDEN
from .memory_utils import create_structured_memory, parse_response

class Node:
    """
    A node represents a discrete task or step in the agent's execution.
    Each node has its own local memory and can access global memory.
    """
    def __init__(self, node_id: Optional[str] = None, task_description: Optional[str] = None, depth: int = 0, node_type: str = "task"):
        """
        Initialize a new Node.
        
        Args:
            node_id: Unique identifier for this node (generated if not provided)
            task_description: Description of the task for this node
            depth: Depth level in the task tree (0 for root)
            node_type: Type of node (task, subtask, etc.)
        """
        self.node_id = node_id if node_id else str(uuid.uuid4())
        self.node_type = node_type
        self.depth = depth
        self.local_memory = LocalMemory(self.node_id)
        self.parent_id: Optional[str] = None
        self.children: List[str] = []
        self.status = "pending"  # pending, in_progress, completed, failed
        
        # Store task description if provided
        if task_description:
            self.store_in_memory("task", task_description)
        
    def store_in_memory(self, key: str, value: Any) -> None:
        """
        Store data in this node's local memory.
        
        Args:
            key: The key to store the value under
            value: The value to store
        """
        self.local_memory.store(key, value)
        
    def retrieve_from_memory(self, key: str) -> Any:
        """
        Retrieve data from this node's local memory.
        
        Args:
            key: The key to retrieve
            
        Returns:
            The stored value, or None if not found
        """
        return self.local_memory.retrieve(key)
    
    def store_llm_response(self, response: str) -> None:
        """
        Process and store an LLM response.
        
        Args:
            response: Raw response from the LLM
        """
        memory_dict = create_structured_memory(response)
        for key, value in memory_dict.items():
            self.store_in_memory(key, value)
    
    def add_child(self, child_id: str) -> None:
        """
        Add a child node ID to this node.
        
        Args:
            child_id: ID of the child node
        """
        if child_id not in self.children:
            self.children.append(child_id)
    
    def set_parent(self, parent_id: str) -> None:
        """
        Set the parent node ID for this node.
        
        Args:
            parent_id: ID of the parent node
        """
        self.parent_id = parent_id
        
    def update_status(self, status: str) -> None:
        """
        Update the status of this node.
        
        Args:
            status: New status (pending, in_progress, completed, failed)
        """
        self.status = status
        self.store_in_memory("status", status)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of this node's state and memory.
        
        Returns:
            Dictionary with node information
        """
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "status": self.status,
            "parent_id": self.parent_id,
            "children": self.children,
            "task": self.retrieve_from_memory("task") or "",
            "result": self.retrieve_from_memory("result") or ""
        }
    
    def save_state(self, directory: str = "node_memory") -> None:
        """
        Save the node's state and memory to disk.
        
        Args:
            directory: Directory to save state to
        """
        self.local_memory.save_to_disk(directory)
    
    def load_state(self, directory: str = "node_memory") -> bool:
        """
        Load the node's state and memory from disk.
        
        Args:
            directory: Directory to load state from
            
        Returns:
            True if loaded successfully, False otherwise
        """
        return self.local_memory.load_from_disk(directory)
