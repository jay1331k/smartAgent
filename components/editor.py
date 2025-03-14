import streamlit as st
import os
import re
import tempfile
import subprocess
from pathlib import Path

class Editor:
    def __init__(self):
        self.current_file = None
        self.content = ""
        
        # Define language-specific features
        self.language_features = {
            'python': {
                'snippets': {
                    'class': 'class MyClass:\n    def __init__(self):\n        pass\n        \n    def my_method(self):\n        pass',
                    'function': 'def my_function(arg1, arg2=None):\n    """\n    Function description\n    \n    Args:\n        arg1: Description\n        arg2: Description\n        \n    Returns:\n        Return value description\n    """\n    return arg1',
                    'if': 'if condition:\n    pass\nelse:\n    pass',
                    'for': 'for item in items:\n    pass',
                    'try': 'try:\n    # code\nexcept Exception as e:\n    print(f"Error: {e}")',
                    'import': 'import module\nfrom module import submodule'
                },
                'extension': '.py',
                'comment': '# '
            },
            'javascript': {
                'snippets': {
                    'function': 'function myFunction(arg1, arg2) {\n    // Function body\n    return arg1;\n}',
                    'arrow': 'const myFunction = (arg1, arg2) => {\n    // Function body\n    return arg1;\n};',
                    'class': 'class MyClass {\n    constructor() {\n        // Constructor\n    }\n    \n    myMethod() {\n        // Method body\n    }\n}',
                    'if': 'if (condition) {\n    // code\n} else {\n    // code\n}',
                    'for': 'for (let i = 0; i < items.length; i++) {\n    const item = items[i];\n    // code\n}',
                    'try': 'try {\n    // code\n} catch (error) {\n    console.error(`Error: ${error}`);\n}'
                },
                'extension': '.js',
                'comment': '// '
            },
            'html': {
                'snippets': {
                    'template': '<!DOCTYPE html>\n<html lang="en">\n<head>\n    <meta charset="UTF-8">\n    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n    <title>Document</title>\n</head>\n<body>\n    \n</body>\n</html>',
                    'div': '<div class="container">\n    \n</div>',
                    'link': '<link rel="stylesheet" href="styles.css">',
                    'script': '<script src="script.js"></script>',
                },
                'extension': '.html',
                'comment': '<!-- -->'
            }
        }
    
    def display(self):
        st.header("💻 Code Editor")
        
        # Add tabs for multiple features
        ide_tabs = st.tabs(["Editor", "AI Assistant", "Documentation"])
        
        with ide_tabs[0]:  # Editor tab
            self._display_editor()
            
        with ide_tabs[1]:  # AI Assistant tab
            self._display_ai_assistant()
            
        with ide_tabs[2]:  # Documentation tab
            self._display_documentation()
    
    def _display_editor(self):
        # If a file is selected from the explorer, display it
        if 'current_file' in st.session_state and 'file_content' in st.session_state:
            self.current_file = st.session_state.current_file
            file_language = st.session_state.get('current_file_language', 'text')
            
            # Show file path and edit options
            st.subheader(f"📄 {os.path.basename(self.current_file)}")
            st.caption(f"Path: {self.current_file}")
            
            # Editor toolbar
            col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
            
            with col1:
                if st.button("Save", key="save_file"):
                    self._save_file()
            
            with col2:
                if st.button("Format", key="format_code"):
                    self._format_code()
            
            with col3:
                if st.button("Run", key="run_code"):
                    self._run_code()
            
            with col4:
                # Code snippets dropdown based on file type
                language = self._detect_language()
                if language and language in self.language_features:
                    snippets = self.language_features[language]['snippets']
                    selected_snippet = st.selectbox(
                        "Insert snippet:", 
                        ["Select snippet..."] + list(snippets.keys()),
                        key="snippet_selector"
                    )
                    
                    if selected_snippet != "Select snippet..." and st.button("Insert"):
                        snippet_content = snippets[selected_snippet]
                        current_content = st.session_state.file_content
                        # Insert at cursor position (not implemented here) or append
                        st.session_state.file_content = current_content + "\n" + snippet_content
                        st.experimental_rerun()
            
            # Text editor with the file content
            # Try to use streamlit-ace if available for better IDE features
            try:
                import streamlit_ace
                
                # Get the appropriate mode for the language
                mode = self._get_ace_mode(file_language)
                
                editor_content = streamlit_ace.st_ace(
                    value=st.session_state.file_content,
                    language=mode,
                    theme="monokai",
                    keybinding="vscode",
                    min_lines=20,
                    max_lines=40,
                    font_size=14,
                    tab_size=4,
                    wrap=True,
                    show_gutter=True,
                    show_print_margin=True,
                    auto_update=True,
                    readonly=False,
                    key="ace_editor"
                )
                
                # Update the session state content if changed
                if editor_content != st.session_state.file_content:
                    st.session_state.file_content = editor_content
                
            except ImportError:
                # Fallback to regular text area if streamlit-ace is not installed
                st.session_state.file_content = st.text_area(
                    "File Editor", 
                    value=st.session_state.file_content,
                    height=400,
                    key="file_editor"
                )
            
            # File info
            file_info_col1, file_info_col2 = st.columns(2)
            with file_info_col1:
                st.info(f"Language: {file_language.capitalize()}")
            with file_info_col2:
                line_count = len(st.session_state.file_content.split('\n'))
                st.info(f"Lines: {line_count}")
            
            # FIXED: Instead of using nested expanders for file preview, use columns or plain text
            st.write("**File Preview:**")
            st.code(st.session_state.file_content, language=file_language)
                
        else:
            # No file selected
            st.info("Select a file from the File Explorer to start editing.")
            
            # Create new file option
            with st.expander("Create New File"):
                language_options = ["Python", "JavaScript", "HTML", "CSS", "Text"]
                selected_language = st.selectbox(
                    "Select language:", 
                    language_options,
                    key="new_file_language"
                )
                
                new_file_path = st.text_input("File path (relative to current directory):")
                
                # Add extension if not provided
                if new_file_path and "." not in new_file_path:
                    extension = self._get_extension_for_language(selected_language.lower())
                    if extension:
                        new_file_path += extension
                
                # Suggest template based on language
                template_content = self._get_template_for_language(selected_language.lower())
                new_file_content = st.text_area(
                    "File content:", 
                    value=template_content,
                    height=200
                )
                
                if st.button("Create File"):
                    if new_file_path:
                        try:
                            # Get the full path
                            full_path = os.path.join(st.session_state.get("explorer_dir", os.getcwd()), new_file_path)
                            
                            # Make sure the directory exists
                            os.makedirs(os.path.dirname(full_path), exist_ok=True)
                            
                            # Create the file
                            with open(full_path, 'w', encoding='utf-8') as f:
                                f.write(new_file_content)
                                
                            st.success(f"File created: {new_file_path}")
                            
                            # Load the file in the editor
                            st.session_state.current_file = full_path
                            st.session_state.file_content = new_file_content
                            file_ext = os.path.splitext(full_path)[1].lower()
                            st.session_state.current_file_language = self._detect_language_from_extension(file_ext)
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Error creating file: {str(e)}")
                    else:
                        st.error("Please specify a file path")
    
    def _display_ai_assistant(self):
        """AI coding assistant functionality"""
        st.subheader("AI Code Assistant")
        
        if 'current_file' in st.session_state and 'file_content' in st.session_state:
            language = self._detect_language()
            
            task_options = [
                "Generate comments for my code",
                "Optimize this code",
                "Add error handling",
                "Explain how this code works",
                "Find potential bugs",
                "Convert to a different language",
                "Custom task"
            ]
            
            selected_task = st.selectbox("Select task:", task_options, key="ai_task")
            
            if selected_task == "Custom task":
                custom_task = st.text_input("Describe what you want the AI to do:")
                if custom_task:
                    selected_task = custom_task
            
            if selected_task == "Convert to a different language":
                target_language = st.selectbox(
                    "Select target language:", 
                    ["Python", "JavaScript", "TypeScript", "Java", "C++", "C#"],
                    key="target_language"
                )
                selected_task = f"Convert to {target_language}"
            
            if st.button("Generate", key="generate_ai"):
                if 'agent' in st.session_state and st.session_state.file_content:
                    with st.spinner("AI is working..."):
                        try:
                            # Create the prompt
                            prompt = f"""
                            Task: {selected_task}
                            
                            Language: {language}
                            
                            Code:
                            ```{language}
                            {st.session_state.file_content}
                            ```
                            
                            Please provide the result in properly formatted code. If you're making changes to the original code,
                            include helpful comments explaining your changes.
                            """
                            
                            # Generate AI response
                            response = st.session_state.agent.llm.generate_content(
                                prompt,
                                generation_config=st.session_state.llm_config
                            )
                            
                            # Extract code from response
                            result = response.text
                            
                            # Display result
                            st.subheader("AI Result")
                            st.markdown(result)
                            
                            # Option to apply changes
                            if st.button("Apply Changes to File"):
                                # Extract code block
                                code_pattern = r"```.*?\n(.*?)```"
                                matches = re.findall(code_pattern, result, re.DOTALL)
                                if matches:
                                    # Use the first code block
                                    new_code = matches[0]
                                    st.session_state.file_content = new_code
                                    st.success("Changes applied to editor")
                                    st.experimental_rerun()
                                else:
                                    st.error("Couldn't extract code from AI response")
                            
                        except Exception as e:
                            st.error(f"Error generating AI response: {str(e)}")
                else:
                    st.error("Please select a file to edit first or ensure the agent is initialized")
        else:
            st.info("Select a file from the File Explorer to use the AI assistant")
    
    def _display_documentation(self):
        """Display language-specific documentation"""
        st.subheader("Documentation")
        
        if 'current_file' in st.session_state:
            language = self._detect_language()
            
            if language == 'python':
                st.markdown("""
                ### Python Resources
                - [Python Official Documentation](https://docs.python.org/3/)
                - [PEP 8 Style Guide](https://peps.python.org/pep-0008/)
                - [Python Standard Library](https://docs.python.org/3/library/index.html)
                - [Python Tutorial](https://docs.python.org/3/tutorial/index.html)
                
                ### Common Libraries
                - [NumPy](https://numpy.org/doc/stable/)
                - [Pandas](https://pandas.pydata.org/docs/)
                - [Matplotlib](https://matplotlib.org/stable/contents.html)
                - [TensorFlow](https://www.tensorflow.org/api_docs)
                - [PyTorch](https://pytorch.org/docs/stable/index.html)
                - [Streamlit](https://docs.streamlit.io/)
                """)
            elif language == 'javascript':
                st.markdown("""
                ### JavaScript Resources
                - [MDN Web Docs](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
                - [JavaScript.info](https://javascript.info/)
                - [ECMAScript Specifications](https://www.ecma-international.org/publications-and-standards/standards/ecma-262/)
                
                ### Common Libraries & Frameworks
                - [React](https://reactjs.org/docs/getting-started.html)
                - [Vue.js](https://vuejs.org/guide/introduction.html)
                - [Angular](https://angular.io/docs)
                - [Node.js](https://nodejs.org/en/docs/)
                - [Express](https://expressjs.com/en/4x/api.html)
                """)
            elif language == 'html':
                st.markdown("""
                ### HTML Resources
                - [MDN HTML Documentation](https://developer.mozilla.org/en-US/docs/Web/HTML)
                - [W3C HTML Specification](https://html.spec.whatwg.org/)
                - [HTML Living Standard](https://html.spec.whatwg.org/multipage/)
                
                ### Related Technologies
                - [CSS Documentation](https://developer.mozilla.org/en-US/docs/Web/CSS)
                - [HTML5 Features](https://developer.mozilla.org/en-US/docs/Web/Guide/HTML/HTML5)
                """)
            else:
                st.write(f"Documentation for {language} will be displayed here.")
        else:
            st.info("Select a file to see relevant documentation")
    
    def _save_file(self):
        """Save the current file"""
        try:
            with open(self.current_file, 'w', encoding='utf-8') as f:
                f.write(st.session_state.file_content)
            st.success(f"File saved: {os.path.basename(self.current_file)}")
        except Exception as e:
            st.error(f"Error saving file: {str(e)}")
    
    def _format_code(self):
        """Format code based on language"""
        language = self._detect_language()
        
        if language == 'python':
            try:
                # Try to use black for Python formatting
                with tempfile.NamedTemporaryFile(suffix=".py", mode='w+', delete=False) as tmp:
                    tmp.write(st.session_state.file_content)
                    tmp_path = tmp.name
                
                try:
                    subprocess.run(["black", tmp_path], check=True, capture_output=True)
                    with open(tmp_path, 'r') as f:
                        formatted_code = f.read()
                    st.session_state.file_content = formatted_code
                    st.success("Code formatted with Black")
                except (subprocess.SubprocessError, FileNotFoundError):
                    # If black fails or isn't installed, use AI formatting
                    self._format_with_ai()
                finally:
                    os.unlink(tmp_path)
                    
            except Exception:
                # Fallback to AI-based formatting if any error occurs
                self._format_with_ai()
        else:
            # For other languages, use AI-based formatting
            self._format_with_ai()
    
    def _run_code(self):
        """Run the current file if possible"""
        if not self.current_file:
            st.error("No file selected")
            return
            
        language = self._detect_language()
        file_path = self.current_file
        
        if language == 'python':
            with st.spinner("Running Python code..."):
                try:
                    result = subprocess.run(
                        ["python", file_path],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    st.subheader("Execution Result")
                    if result.stdout:
                        st.text("Output:")
                        st.code(result.stdout)
                    if result.stderr:
                        st.text("Errors:")
                        st.error(result.stderr)
                    
                    st.info(f"Process exited with code {result.returncode}")
                    
                except subprocess.TimeoutExpired:
                    st.error("Execution timed out (10s limit)")
                except Exception as e:
                    st.error(f"Error executing code: {str(e)}")
                    
        elif language == 'javascript' and os.path.exists("node"):
            with st.spinner("Running JavaScript code with Node.js..."):
                try:
                    result = subprocess.run(
                        ["node", file_path],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    st.subheader("Execution Result")
                    if result.stdout:
                        st.text("Output:")
                        st.code(result.stdout)
                    if result.stderr:
                        st.text("Errors:")
                        st.error(result.stderr)
                    
                    st.info(f"Process exited with code {result.returncode}")
                    
                except subprocess.TimeoutExpired:
                    st.error("Execution timed out (10s limit)")
                except Exception as e:
                    st.error(f"Error executing code: {str(e)}")
        else:
            st.warning(f"Running {language} files not supported in the IDE")
    
    def _format_with_ai(self):
        """Format code using the AI model"""
        if 'agent' in st.session_state:
            with st.spinner("Formatting code..."):
                try:
                    language = self._detect_language()
                    prompt = f"""
                    Please format the following code according to best practices for {language}.
                    Return only the formatted code without any explanation.
                    
                    ```{language}
                    {st.session_state.file_content}
                    ```
                    """
                    
                    response = st.session_state.agent.llm.generate_content(
                        prompt,
                        generation_config={**st.session_state.llm_config, "temperature": 0.1}
                    )
                    
                    # Extract code from response
                    code_pattern = r"```.*?\n(.*?)```"
                    matches = re.findall(code_pattern, response.text, re.DOTALL)
                    if matches:
                        formatted_code = matches[0]
                        st.session_state.file_content = formatted_code
                        st.success("Code formatted successfully")
                    else:
                        st.session_state.file_content = response.text.strip()
                        st.warning("AI formatting response didn't match expected format")
                    
                except Exception as e:
                    st.error(f"Error formatting code: {str(e)}")
        else:
            st.error("Agent not initialized. Cannot format code.")
    
    def _detect_language(self) -> str:
        """Detect language based on file extension"""
        if not self.current_file:
            return "text"
        
        file_ext = os.path.splitext(self.current_file)[1].lower()
        return self._detect_language_from_extension(file_ext)
    
    def _detect_language_from_extension(self, file_ext: str) -> str:
        """Detect language from file extension"""
        extension_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.html': 'html',
            '.css': 'css',
            '.json': 'json',
            '.md': 'markdown',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php'
        }
        return extension_map.get(file_ext, 'text')
    
    def _get_ace_mode(self, language: str) -> str:
        """Convert language to ace editor mode"""
        # Ace editor uses slightly different mode names than our internal ones
        mode_map = {
            'python': 'python',
            'javascript': 'javascript',
            'typescript': 'typescript',
            'html': 'html',
            'css': 'css',
            'json': 'json',
            'markdown': 'markdown',
            'java': 'java',
            'cpp': 'c_cpp',
            'c': 'c_cpp',
            'csharp': 'csharp',
            'go': 'golang',
            'rust': 'rust',
            'php': 'php'
        }
        return mode_map.get(language, 'text')
    
    def _get_extension_for_language(self, language: str) -> str:
        """Get file extension for language"""
        extension_map = {
            'python': '.py',
            'javascript': '.js',
            'typescript': '.ts',
            'html': '.html',
            'css': '.css',
            'json': '.json',
            'markdown': '.md',
            'java': '.java',
            'cpp': '.cpp',
            'c': '.c',
            'csharp': '.cs',
            'go': '.go',
            'rust': '.rs',
            'php': '.php'
        }
        return extension_map.get(language, '.txt')
    
    def _get_template_for_language(self, language: str) -> str:
        """Get template for new file based on language"""
        templates = {
            'python': '#!/usr/bin/env python3\n\n"""\nDescription: \n\nAuthor: \nDate: \n"""\n\n\ndef main():\n    print("Hello, world!")\n\n\nif __name__ == "__main__":\n    main()\n',
            'javascript': '/**\n * Description: \n * \n * @author \n */\n\nconst main = () => {\n    console.log("Hello, world!");\n};\n\nmain();\n',
            'html': '<!DOCTYPE html>\n<html lang="en">\n<head>\n    <meta charset="UTF-8">\n    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n    <title>Document</title>\n</head>\n<body>\n    <h1>Hello, world!</h1>\n</body>\n</html>',
            'css': '/* \n * Description: \n * Author: \n */\n\nbody {\n    font-family: Arial, sans-serif;\n    margin: 0;\n    padding: 0;\n    background-color: #f0f0f0;\n}\n',
        }
        return templates.get(language, '')
