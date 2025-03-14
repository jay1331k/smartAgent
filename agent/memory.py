"""Memory modules for storing and retrieving agent information."""

from typing import Dict, Any, Optional

class LocalMemory:
    """Memory specific to a node, storing task-specific information."""
    
    def __init__(self, node_id: str):
        """Initialize local memory for a specific node.
        
        Args:
            node_id: The ID of the node this memory belongs to
        """
        self.node_id = node_id
        self.local_memory: Dict[str, Any] = {}
        
    def store(self, key: str, value: Any) -> None:
        """Store a value in memory.
        
        Args:
            key: The key to store the value under
            value: The value to store
        """
        self.local_memory[key] = value
        
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value from memory.
        
        Args:
            key: The key to retrieve
            
        Returns:
            The stored value, or None if not found
        """
        return self.local_memory.get(key)
        
    def clear(self) -> None:
        """Clear all memory."""
        self.local_memory = {}
        
    def get_all(self) -> Dict[str, Any]:
        """Get all memory contents.
        
        Returns:
            Dict containing all stored memory
        """
        return self.local_memory

    def save_to_disk(self, directory: str = "node_memory") -> None:
        """Save memory to disk.
        
        Args:
            directory: Directory to save to
        """
        import os
        import json
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, f"{self.node_id}.json")
        with open(filepath, 'w') as f:
            json.dump(self.local_memory, f, indent=2)
    
    def load_from_disk(self, directory: str = "node_memory") -> bool:
        """Load memory from disk.
        
        Args:
            directory: Directory to load from
            
        Returns:
            True if successful, False otherwise
        """
        import os
        import json
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
    """Global memory shared across all nodes."""
    
    def __init__(self):
        """Initialize global memory."""
        self.global_context: str = "This agent can solve tasks by breaking them down into subtasks."
        self.shared_data: Dict[str, Any] = {}
        
    def update_context(self, context: str) -> None:
        """Update the global context.
        
        Args:
            context: New context string
        """
        self.global_context = context
        
    def get_context(self) -> str:
        """Get the current global context.
        
        Returns:
            The current context string
        """
        return self.global_context
        
    def store(self, key: str, value: Any) -> None:
        """Store a value in shared memory.
        
        Args:
            key: The key to store the value under
            value: The value to store
        """
        self.shared_data[key] = value
        
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value from shared memory.
        
        Args:
            key: The key to retrieve
            
        Returns:
            The stored value, or None if not found
        """
        return self.shared_data.get(key)

    def save_to_disk(self, filepath: str = "global_memory.json") -> None:
        """Save global memory to disk.
        
        Args:
            filepath: Path to save to
        """
        import json
        with open(filepath, 'w') as f:
            json.dump({
                "global_context": self.global_context,
                "shared_data": self.shared_data
            }, f, indent=2)
    
    def load_from_disk(self, filepath: str = "global_memory.json") -> bool:
        """Load global memory from disk.
        
        Args:
            filepath: Path to load from
            
        Returns:
            True if successful, False otherwise
        """
        import os
        import json
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
