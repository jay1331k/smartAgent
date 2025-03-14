#!/usr/bin/env python
"""
Diagnostic script to test imports and basic functionality
This can help diagnose issues with imports or dependencies
"""
import os
import sys
import importlib.util
from pathlib import Path
import tempfile

def check_file_exists(file_path):
    """Check if a file exists and return its contents if it does."""
    path = Path(file_path)
    if path.is_file():
        print(f"✓ File exists: {file_path}")
        return True
    else:
        print(f"❌ File does not exist: {file_path}")
        return False

def test_import(module_name, class_name=None):
    """Test importing a module and optionally a class from it."""
    try:
        module = __import__(module_name)
        print(f"✓ Successfully imported {module_name}")
        
        if class_name:
            try:
                # Try to access the class
                class_obj = getattr(module, class_name)
                print(f"✓ Successfully imported {class_name} from {module_name}")
                return True
            except AttributeError:
                print(f"❌ Could not find {class_name} in {module_name}")
                return False
        return True
    except ImportError as e:
        print(f"❌ Failed to import {module_name}: {str(e)}")
        return False

def main():
    """Run diagnostic tests."""
    print("\n=== Diagnosing Cline Web IDE imports ===\n")
    
    # Check if key files exist
    files_to_check = [
        "app.py",
        "file_explorer.py",
        "terminal.py",
        "cline_interface.py",
        "requirements.txt"
    ]
    
    all_files_exist = True
    for file in files_to_check:
        if not check_file_exists(file):
            all_files_exist = False
    
    if not all_files_exist:
        print("\n❌ Some required files are missing. Please check the file structure.")
        return
    
    # Test imports
    print("\n--- Testing imports ---\n")
    imports_to_test = [
        ("file_explorer", "FileExplorer"),
        ("terminal", "Terminal"),
        ("cline_interface", "ClineInterface")
    ]
    
    all_imports_ok = True
    for module_name, class_name in imports_to_test:
        if not test_import(module_name, class_name):
            all_imports_ok = False
    
    if not all_imports_ok:
        print("\n❌ Some imports failed. Please check the import statements and class definitions.")
        
        # Try to provide more information about the failing imports
        print("\n--- Detailed module inspection ---")
        for module_name, class_name in imports_to_test:
            try:
                spec = importlib.util.find_spec(module_name)
                if spec is not None:
                    print(f"Module {module_name} found at: {spec.origin}")
                    
                    # Try to inspect module contents
                    try:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        module_contents = dir(module)
                        print(f"Contents of {module_name}: {module_contents}")
                        
                        if class_name not in module_contents:
                            print(f"❌ Class {class_name} not found in {module_name}")
                            
                            # Check if there's a similar class name (typo?)
                            possible_classes = [name for name in module_contents if name.lower() == class_name.lower()]
                            if possible_classes:
                                print(f"   Did you mean: {possible_classes[0]}?")
                    except Exception as e:
                        print(f"Error inspecting module: {str(e)}")
                else:
                    print(f"Could not find module spec for {module_name}")
            except Exception as e:
                print(f"Error inspecting {module_name}: {str(e)}")
    else:
        print("\n✓ All imports successful!")
        
        print("\nTrying to initialize the main components...")
        try:
            # Try to create instances
            from file_explorer import FileExplorer
            from terminal import Terminal
            from cline_interface import ClineInterface
            
            temp_dir = tempfile.mkdtemp(prefix="cline_test_")
            file_explorer = FileExplorer(temp_dir)
            terminal = Terminal(temp_dir)
            cline_interface = ClineInterface()
            
            print("✓ Successfully initialized components!")
            print(f"\nTemp directory for testing: {temp_dir}")
        except Exception as e:
            print(f"❌ Error initializing components: {str(e)}")

if __name__ == "__main__":
    main()
