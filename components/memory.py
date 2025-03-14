import os
import json
from typing import Optional, Dict, List, Any, Union, Tuple

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
