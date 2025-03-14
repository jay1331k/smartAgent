import os
import json
from typing import Dict, Any, Optional

class GlobalMemory:
    """
    Manages global memory for the agent, including context and shared data.
    """
    def __init__(self, initial_context: str = ""):
        self.global_context = initial_context
        self.shared_data = {}
        
    def update_context(self, new_context: str) -> None:
        """Update the global context."""
        self.global_context = new_context
        
    def get_context(self) -> str:
        """Get the current global context."""
        return self.global_context
        
    def store(self, key: str, value: Any) -> None:
        """Store data in global shared memory."""
        self.shared_data[key] = value
        
    def retrieve(self, key: str) -> Any:
        """Retrieve data from global shared memory."""
        return self.shared_data.get(key)
        
    def clear(self) -> None:
        """Clear all global shared memory."""
        self.shared_data.clear()
        
    def save_to_disk(self, filename: str = "global_memory.json") -> bool:
        """Save global memory to disk."""
        try:
            data = {
                "global_context": self.global_context,
                "shared_data": self.shared_data
            }
            with open(filename, "w") as f:
                json.dump(data, f, indent=2)
            return True
        except Exception:
            return False
            
    def load_from_disk(self, filename: str = "global_memory.json") -> bool:
        """Load global memory from disk."""
        try:
            if not os.path.exists(filename):
                return False
                
            with open(filename, "r") as f:
                data = json.load(f)
                
            self.global_context = data.get("global_context", "")
            self.shared_data = data.get("shared_data", {})
            return True
        except Exception:
            return False

class LocalMemory:
    """
    Manages local memory for a specific node, allowing data storage and retrieval.
    """
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.local_memory: Dict[str, Any] = {}
        
    def store(self, key: str, value: Any) -> None:
        """Store data in local memory."""
        self.local_memory[key] = value
        
    def retrieve(self, key: str) -> Any:
        """Retrieve data from local memory."""
        return self.local_memory.get(key)
        
    def clear(self) -> None:
        """Clear all local memory."""
        self.local_memory.clear()
        
    def get_keys(self) -> list:
        """Get all keys in local memory."""
        return list(self.local_memory.keys())
        
    def get_all(self) -> Dict[str, Any]:
        """Get all data in local memory."""
        return self.local_memory.copy()
        
    def save_to_disk(self, directory: str = "node_memory") -> bool:
        """Save local memory to disk."""
        try:
            os.makedirs(directory, exist_ok=True)
            filename = os.path.join(directory, f"{self.node_id}.json")
            
            with open(filename, "w") as f:
                json.dump(self.local_memory, f, indent=2)
            return True
        except Exception:
            return False
            
    def load_from_disk(self, directory: str = "node_memory") -> bool:
        """Load local memory from disk."""
        try:
            filename = os.path.join(directory, f"{self.node_id}.json")
            
            if not os.path.exists(filename):
                return False
                
            with open(filename, "r") as f:
                self.local_memory = json.load(f)
            return True
        except Exception:
            return False
