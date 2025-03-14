"""
Launcher for the SmartAgent application.
"""
import os
import sys
import argparse

def run_streamlit():
    """Run the Streamlit web interface."""
    import subprocess
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Change to the script directory
    os.chdir(script_dir)
    
    # Run the Streamlit app
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port=8501"]
    
    try:
        print("SmartAgent is running at http://localhost:8501")
        print("Press Ctrl+C to stop")
        process = subprocess.Popen(cmd)
        process.wait()
    except KeyboardInterrupt:
        print("\nStopping SmartAgent...")
        process.terminate()
    except Exception as e:
        print(f"Error: {e}")

def run_cli():
    """Run the command-line interface."""
    from agent.node import Node
    from agent.memory import GlobalMemory
    from agent.memory_utils import parse_response
    import json
    import uuid
    from typing import Dict, Optional, Any, List
    
    # Initialize global memory and nodes
    global_memory = GlobalMemory()
    nodes: Dict[str, Node] = {}
    
    # Load existing state if available
    if os.path.exists("global_memory.json"):
        global_memory.load_from_disk()
        
    if os.path.exists("node_memory"):
        for filename in os.listdir("node_memory"):
            if filename.endswith(".json"):
                node_id = filename.replace(".json", "")
                new_node = Node(node_id=node_id)
                if new_node.load_state():
                    nodes[node_id] = new_node
    
    # Create root node if needed
    if "root" not in nodes:
        root_node = Node(node_id="root", task_description="Root task", depth=0, node_type="root")
        root_node.update_status("completed")
        nodes["root"] = root_node
    
    print("Welcome to SmartAgent CLI")
    print("=" * 50)
    
    # CLI menu loop
    while True:
        print("\nMAIN MENU")
        print("1. View task tree")
        print("2. View node details")
        print("3. Create new task")
        print("4. Process task with LLM response")
        print("5. Save agent state")
        print("6. Load agent state")
        print("0. Exit")
        
        choice = input("Enter choice: ")
        
        # Handle menu choices
        # ...existing code...

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="Run the SmartAgent application")
    parser.add_argument("--cli", action="store_true", help="Run in command-line interface mode")
    args = parser.parse_args()
    
    if args.cli:
        run_cli()
    else:
        run_streamlit()

if __name__ == "__main__":
    main()
