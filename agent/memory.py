"""
Memory classes for SmartAgent.
"""
import os
import json
from typing import Dict, Optional, Any, Union

class LocalMemory:
    """
    Local memory for a specific node.
    Stores key-value pairs related to a single node's execution.
    """
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.local_memory: Dict[str, Any] = {}
    
    def store(self, key: str, value: Any) -> None:
        """
        Store a value in local memory.
        
        Args:
            key: The key to store the value under
            value: The value to store (can be any JSON-serializable type)
        """
        self.local_memory[key] = value
    
    def retrieve(self, key: str) -> Optional[Any]:
        """
        Retrieve a value from local memory.
        
        Args:
            key: The key to retrieve
            
        Returns:
            The value, or None if not found
        """
        return self.local_memory.get(key)
    
    def clear(self) -> None:
        """Clear all local memory."""
        self.local_memory = {}
        
    def get_all(self) -> Dict[str, Any]:
        """
        Get all stored memory items.
        
        Returns:
            Dictionary containing all memory items
        """
        return self.local_memory
        
    def save_to_disk(self, directory: str = "node_memory") -> None:
        """
        Save local memory to disk.
        
        Args:
            directory: Directory to save memory to
        """
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, f"{self.node_id}.json")
        with open(filepath, 'w') as f:
            json.dump(self.local_memory, f, indent=2)
    
    def load_from_disk(self, directory: str = "node_memory") -> bool:
        """
        Load local memory from disk.
        
        Args:
            directory: Directory to load memory from
            
        Returns:
            True if loaded successfully, False otherwise
        """
        filepath = os.path.join(directory, f"{self.node_id}.json")
        if not os.path.exists(filepath):
            return False
            
        try:
            with open(filepath, 'r') as f:
                self.local_memory = json.load(f)
            return True
        except Exception:
            return False

class GlobalMemory:
    """
    Global memory for the agent.
    Stores information that needs to be accessible across all nodes.
    """
    def __init__(self):
        self.global_context: str = "This agent can solve tasks by breaking them down into subtasks."
        self.shared_data: Dict[str, Any] = {}
    
    def update_context(self, context: str) -> None:
        """
        Update the global context.
        
        Args:
            context: The new context
        """
        self.global_context = context
    
    def get_context(self) -> str:
        """
        Get the current global context.
        
        Returns:
            The global context
        """
        return self.global_context
    
    def store(self, key: str, value: Any) -> None:
        """
        Store a value in global memory.
        
        Args:
            key: The key to store the value under
            value: The value to store
        """
        self.shared_data[key] = value
    
    def retrieve(self, key: str) -> Optional[Any]:
        """
        Retrieve a value from global memory.
        
        Args:
            key: The key to retrieve
            
        Returns:
            The value, or None if not found
        """
        return self.shared_data.get(key)
    
    def save_to_disk(self, filepath: str = "global_memory.json") -> None:
        """
        Save global memory to disk.
        
        Args:
            filepath: The path to save to
        """
        with open(filepath, 'w') as f:
            json.dump({
                "global_context": self.global_context,
                "shared_data": self.shared_data
            }, f, indent=2)
    
    def load_from_disk(self, filepath: str = "global_memory.json") -> bool:
        """
        Load global memory from disk.
        
        Args:
            filepath: The path to load from
            
        Returns:
            True if loaded successfully, False otherwise
        """
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
