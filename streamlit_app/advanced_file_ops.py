import os
import shutil
import streamlit as st
from pathlib import Path
import tempfile
import mimetypes
import zipfile
import io

class AdvancedFileOperations:
    """Advanced file operations for the IDE."""
    
    def __init__(self, root_path):
        self.root_path = Path(root_path)
        
    def delete_file(self, file_path):
        """Delete a file or directory."""
        try:
            path = Path(file_path)
            if not path.exists():
                return False, f"Path does not exist: {path}"
            
            if path.is_file():
                path.unlink()
                return True, f"File deleted: {path.name}"
            elif path.is_dir():
                shutil.rmtree(path)
                return True, f"Directory deleted: {path.name}"
                
        except Exception as e:
            return False, f"Error deleting file: {str(e)}"
    
    def rename_file(self, old_path, new_name):
        """Rename a file or directory."""
        try:
            old_path = Path(old_path)
            if not old_path.exists():
                return False, f"Path does not exist: {old_path}"
                
            new_path = old_path.parent / new_name
            if new_path.exists():
                return False, f"{new_name} already exists"
                
            old_path.rename(new_path)
            return True, f"Renamed to {new_name}"
                
        except Exception as e:
            return False, f"Error renaming file: {str(e)}"
    
    def move_file(self, source_path, dest_dir):
        """Move a file or directory to another directory."""
        try:
            source = Path(source_path)
            dest = Path(dest_dir)
            
            if not source.exists():
                return False, f"Source does not exist: {source}"
                
            if not dest.exists() or not dest.is_dir():
                return False, f"Destination directory does not exist: {dest}"
                
            new_path = dest / source.name
            if new_path.exists():
                return False, f"A file with the same name already exists in the destination"
                
            shutil.move(str(source), str(dest))
            return True, f"Moved {source.name} to {dest.name}/"
                
        except Exception as e:
            return False, f"Error moving file: {str(e)}"
    
    def copy_file(self, source_path, dest_dir):
        """Copy a file or directory to another directory."""
        try:
            source = Path(source_path)
            dest = Path(dest_dir)
            
            if not source.exists():
                return False, f"Source does not exist: {source}"
                
            if not dest.exists() or not dest.is_dir():
                return False, f"Destination directory does not exist: {dest}"
                
            new_path = dest / source.name
            if new_path.exists():
                return False, f"A file with the same name already exists in the destination"
                
            if source.is_file():
                shutil.copy2(str(source), str(dest))
            else:
                shutil.copytree(str(source), str(dest / source.name))
                
            return True, f"Copied {source.name} to {dest.name}/"
                
        except Exception as e:
            return False, f"Error copying file: {str(e)}"
    
    def create_file(self, path, content=""):
        """Create a new file with the given content."""
        try:
            file_path = Path(path)
            
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create the file
            with open(file_path, 'w') as f:
                f.write(content)
                
            return True, f"File created: {file_path.name}"
                
        except Exception as e:
            return False, f"Error creating file: {str(e)}"
    
    def create_directory(self, path):
        """Create a new directory."""
        try:
            dir_path = Path(path)
            dir_path.mkdir(parents=True, exist_ok=True)
            return True, f"Directory created: {dir_path.name}"
                
        except Exception as e:
            return False, f"Error creating directory: {str(e)}"
    
    def import_from_zip(self, zip_file):
        """Import files from a zip archive."""
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                # Check if the zip has a single root folder
                root_folders = {item.split('/')[0] for item in zip_ref.namelist() 
                               if not item.startswith('__MACOSX')}
                
                extract_path = self.root_path
                zip_ref.extractall(extract_path)
                
            return True, f"Imported files from {zip_file.name}"
                
        except Exception as e:
            return False, f"Error importing from zip: {str(e)}"
    
    def export_to_zip(self, directory=None):
        """Export a directory (or the entire project) to a zip file."""
        try:
            if directory is None:
                directory = self.root_path
            else:
                directory = Path(directory)
                
            if not directory.exists() or not directory.is_dir():
                return False, None, "Directory does not exist"
                
            # Create a memory file to store the zip
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(directory):
                    for file in files:
                        file_path = Path(root) / file
                        zipf.write(
                            file_path, 
                            Path(root).relative_to(directory) / file
                        )
            
            zip_buffer.seek(0)
            return True, zip_buffer, f"Directory {directory.name} exported successfully"
                
        except Exception as e:
            return False, None, f"Error exporting to zip: {str(e)}"
