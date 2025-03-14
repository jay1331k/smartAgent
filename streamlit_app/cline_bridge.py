import os
import sys
import json
from pathlib import Path

# Add the parent directory to sys.path to import from the original Cline project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core Cline functionality
try:
    from src.core.Cline import Cline
    from src.shared.api import ApiConfiguration
    from src.shared.AutoApprovalSettings import AutoApprovalSettings
    from src.shared.BrowserSettings import BrowserSettings
    from src.shared.ChatSettings import ChatSettings
    import src.services.auth.config as auth_config
    CLINE_IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import Cline modules: {e}")
    CLINE_IMPORTS_AVAILABLE = False

class ClineBridge:
    """Bridge between Streamlit app and original Cline functionality."""
    
    def __init__(self, project_path):
        self.project_path = project_path
        self.cline_instance = None
        self.provider_mock = None
        
    def initialize_cline(self, api_key, model_id="claude-3-sonnet-20240229"):
        """Initialize a Cline instance with the given API key."""
        if not CLINE_IMPORTS_AVAILABLE:
            raise ImportError("Cline modules are not available. Cannot initialize.")
        
        # Create a mock provider that fulfills the minimum requirements
        class ProviderMock:
            def __init__(self, context_path):
                self.context = type('obj', (object,), {
                    'globalStorageUri': type('obj', (object,), {
                        'fsPath': context_path
                    })
                })
                
            def deref(self):
                return self
                
            def postStateToWebview(self):
                pass
                
            def postMessageToWebview(self, message):
                pass
                
            def mcpHub(self):
                return None
        
        # Create storage directory for Cline
        storage_path = Path(self.project_path) / ".cline_storage"
        storage_path.mkdir(exist_ok=True)
        
        self.provider_mock = ProviderMock(str(storage_path))
        
        # Create API configuration
        api_config = ApiConfiguration(
            apiProvider="anthropic",
            anthropicApiKey=api_key,
            openAiApiKey="",
            openAiModelId="",
            vsCodeLmModelSelector="",
            ollamaModelId="",
            lmStudioModelId="",
            liteLlmModelId="",
            requestyModelId="",
            openAiModelInfo=None
        )
        
        # Create settings
        auto_approval_settings = AutoApprovalSettings(
            enabled=True,
            maxRequests=10,
            enableNotifications=False
        )
        
        browser_settings = BrowserSettings(
            headless=True,
            slowMo=50
        )
        
        chat_settings = ChatSettings(
            mode="act"
        )
        
        # Create Cline instance
        self.cline_instance = Cline(
            provider=self.provider_mock,
            apiConfiguration=api_config,
            autoApprovalSettings=auto_approval_settings,
            browserSettings=browser_settings,
            chatSettings=chat_settings,
            task="Assist with development in Streamlit Web IDE"
        )
        
        return self.cline_instance
    
    def execute_task(self, message):
        """Execute a task with Cline."""
        if not self.cline_instance:
            raise ValueError("Cline instance not initialized")
        
        # Convert message to Anthropic format
        user_content = [{
            "type": "text", 
            "text": message
        }]
        
        # Execute the task and get response
        self.cline_instance.recursivelyMakeClineRequests(
            userContent=user_content,
            includeFileDetails=True,
            isNewTask=False
        )
        
        # Get the latest assistant message
        assistant_messages = [m for m in self.cline_instance.clineMessages 
                             if m.get("type") == "say" and m.get("say") == "text"]
        
        if assistant_messages:
            return assistant_messages[-1].get("text", "")
        else:
            return "No response from Cline"
    
    def get_firebase_config(self):
        """Get Firebase configuration from original Cline project."""
        if CLINE_IMPORTS_AVAILABLE:
            return auth_config.firebaseConfig
        else:
            # Return default config if imports not available
            return {
                "apiKey": "AIzaSyDcXAaanNgR2_T0dq2oOl5XyKPksYHppVo",
                "authDomain": "cline-bot.firebaseapp.com",
                "projectId": "cline-bot",
                "storageBucket": "cline-bot.firebasestorage.app",
                "messagingSenderId": "364369702101",
                "appId": "1:364369702101:web:0013885dcf20b43799c65c",
                "measurementId": "G-MDPRELSCD1",
            }
