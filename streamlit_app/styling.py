import streamlit as st

def apply_custom_css():
    """Apply custom CSS styling to the Streamlit app."""
    st.markdown("""
    <style>
        /* Main interface styling */
        .main {
            background-color: #1e1e1e;
            color: #e0e0e0;
        }
        
        /* Code editor styling */
        .stTabs {
            background-color: #252526;
            border-radius: 5px;
        }
        .stTextInput > div > div > input {
            background-color: #3c3c3c;
            color: #cccccc;
        }
        
        /* File explorer styling */
        .file-explorer {
            background-color: #252526;
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 10px;
        }
        
        /* Terminal styling */
        .terminal {
            font-family: 'Courier New', monospace;
            background-color: #1e1e1e;
            color: #e0e0e0;
            border: 1px solid #3c3c3c;
            border-radius: 5px;
        }
        
        /* UI components */
        .stButton>button {
            background-color: #0078d7;
            color: white;
        }
        .stButton>button:hover {
            background-color: #0063b1;
        }
        
        /* Chat interface styling */
        .chat-message {
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
        }
        .chat-message-user {
            background-color: #2d2d2d;
        }
        .chat-message-assistant {
            background-color: #3c3c3c;
        }
        
        /* Override Streamlit's default padding */
        .css-18e3th9 {
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #252526;
        }
        
        /* Improve the look of forms */
        .stForm {
            background-color: #2d2d2d;
            padding: 10px;
            border-radius: 5px;
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #1e1e1e;
        }
        ::-webkit-scrollbar-thumb {
            background: #555;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #777;
        }
        
        /* Ace editor min height */
        .ace_editor {
            min-height: 600px !important;
        }
    </style>
    """, unsafe_allow_html=True)

def custom_component_styles():
    """Return CSS classes for custom styling of components."""
    return {
        "file_explorer": "file-explorer",
        "terminal": "terminal",
        "chat_message_user": "chat-message chat-message-user",
        "chat_message_assistant": "chat-message chat-message-assistant",
    }
