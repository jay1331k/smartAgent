# Hierarchical Task Decomposition Agent (Human-Guided)

This project implements a hierarchical task decomposition agent using Streamlit, Google's Gemini API, and a custom architecture inspired by the "Tree of Thoughts" concept.  The agent breaks down complex tasks into smaller, manageable sub-tasks, forming a dynamic tree structure.  Crucially, the entire process is *human-guided*, meaning that each step requires explicit user approval before execution.

## Features

*   **Hierarchical Task Decomposition:** The agent recursively breaks down tasks into sub-tasks, creating a tree-like structure.
*   **Human-in-the-Loop:**  Every node execution requires explicit user interaction (clicking an "Execute Node" button).  This provides fine-grained control and prevents the agent from running autonomously.
*   **Interactive Visualization:**  A dynamic, interactive Plotly graph visualizes the task decomposition tree, allowing users to:
    *   Select nodes to view details.
    *   See node status (pending, running, completed, failed, overridden) through color-coding.
    *   Understand dependencies between nodes.
    *   Highlight the selected node, its parent, and its children.
    *   View detailed tooltips with task descriptions and output snippets.
*   **Textual Representation:**  A textual representation of the node hierarchy complements the graph, providing an alternative view.
*   **Node Details Panel:** Displays comprehensive information about the selected node:
    *   Task description
    *   Status
    *   Constraints
    *   Dependencies
    *   Output
    *   Error messages (if any)
    *   Local Memory contents
*   **Action Buttons:**
    *   **Execute Node:** Executes the selected node (only available when the node is in the "pending" state).
    *   **Regenerate Node:** Resets a node to the "pending" state, allowing the user to provide additional guidance for re-execution.
    *   **Delete Node:** Deletes a node and all its children (with a confirmation dialog).
*   **Undo Functionality:** Reverts the agent to its previous state (before the last node execution, regeneration, or deletion).
*   **Session Management:**
    *   **Save Session:** Saves the current state of the agent (node hierarchy, attention mechanism, global memory, execution count) to a JSON file.
    *   **Load Session:** Loads a previously saved session.
*   **Global Memory:**  A shared memory space that stores a summary of the overall task and the results of completed nodes. Users can view the full global memory content.
*   **Local Memory:** Each node has its own local memory to store intermediate results and maintain context.
*   **Constraint System:**  Allows users to specify constraints on the LLM's output (format, content, length).
*   **Dependency Tracking:**  The agent tracks dependencies between nodes, ensuring that a node is executed only after its dependencies are met.
*   **Error Handling:** Includes retry mechanisms for LLM API errors (rate limits, timeouts) and provides clear error messages to the user.
* **Step Count** Provides total steps executed by the agent.

## Project Structure

The project is organized into the following files:

*   **`ui/app.py`:** The main Streamlit application file.  This contains the user interface and interacts with the `Agent` class.
*   **`agent/agent.py`:** Defines the `Agent` class, which manages the overall task decomposition process, including node creation, execution, regeneration, deletion, and the `agentFlow` function.  Also defines the project's constants.
*   **`agent/node.py`:** Defines the `Node` class, representing a single node in the task decomposition tree.  Handles prompt building, LLM output processing, and local memory.
*   **`agent/memory.py`:** Defines the `LocalMemory` and `GlobalMemory` classes for managing node-specific and global information.
*   **`agent/attention_mechanism.py`:** Defines the `AttentionMechanism` class, responsible for tracking dependencies between nodes, enforcing constraints, and managing the global context.
*   **`agent/utils.py`:** Contains utility functions, including the Plotly graph generation, textual node display, error handling for LLM API retries, and constraint parsing.
*   **`requirements.txt`:** Lists the required Python packages.
*   **`.env`:** Stores environment variables, including your Gemini API key.
*   **`agent_memory.json`:** A temporary file used to store the local memory of nodes. This file is created/deleted during runtime.
* **`session.json`** The file which contains the saved session.

## Getting Started

1.  **Clone the Repository:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up the Environment:**

    *   Create a `.env` file in the root directory of the project.
    *   Add your Google Gemini API key to the `.env` file:

        ```
        GEMINI_API_KEY=your_gemini_api_key_here
        ```
     Replace `your_gemini_api_key_here` with your actual API key.

4.  **Run the Application:**

    ```bash
    streamlit run ui/app.py
    ```

    This will open the application in your web browser.

## Usage

1.  **Enter the Initial Task:**  Type the overall task you want the agent to perform in the "Enter the initial task" text box.
2.  **Add Constraints (Optional):** Use the "Add Constraints" expander to specify any constraints on the LLM's output. You can choose from "format," "contains," and "max_length" constraint types.
3.  **Start the Agent:** Click the "Start Agent" button. This creates the root node of the task decomposition tree.
4.  **Execute Nodes:**  The graph and textual representation will show the node hierarchy.  Select a node to view its details.  If a node is in the "pending" state, you can click the "Execute Node" button to execute it.
5.  **Regenerate Nodes:** If you're not satisfied with a node's output, you can click "Regenerate Node" to reset it to the "pending" state. You can optionally provide additional guidance.
6.  **Delete Nodes:**  Click "Delete Node" to remove a node and its children (after confirming the deletion).
7.  **Undo:** Click "Undo" to revert to the previous state.
8.  **View Global Memory:** Click "View Global Memory" to see the current global context.
9.  **Save/Load Sessions:** Use the "Save Session" and "Load Session" buttons to save and load the agent's state.

## Analogy

The system is like a team of generalist experts who can all break down problems, working together under the guidance of a project manager. They have individual notebooks and a shared whiteboard, allowing them to stay organized and focused, even on very complex projects. The homogeneous structure makes the team easier to manage and more adaptable.

## Future Improvements

*   **Onboarding/Tutorial:**  Add an interactive tutorial to guide new users through the application.
*   **Simplified Initial View:**  Hide advanced features (like the dependency graph) by default, providing an option to show an "Advanced View."
*   **Preset Constraints:** Offer pre-defined constraints for common scenarios.
*   **Visual Constraint Editor:**  Implement a more visual way to define constraints.
*   **Dependency Highlighting:**  Highlight dependency chains in the graph more clearly.
*   **Global Context Editing:**  *Carefully* consider allowing users to edit the global context directly.
*   **More Granular Undo/Redo:** Track individual actions for finer-grained undo/redo capabilities.
*   **Multiple Session Files:** Allow users to save and load multiple sessions with different names.
*   **Structured Task Input:**  Provide a more structured way to input the initial task, potentially with separate fields for goals, inputs, and expected outputs.
* **Example Tasks:** Provide example task for users to start from.

This README provides a comprehensive overview of the project, its features, how to set it up, and how to use it. It also includes an analogy to aid understanding and outlines potential future improvements. This makes the project much more accessible and understandable for other developers and users.
