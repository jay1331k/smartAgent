import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth
import json
import os
from pathlib import Path

# Import the Firebase configuration from the original Cline project or config loader
try:
    from cline_bridge import ClineBridge
    bridge = ClineBridge(".")
    firebase_config = bridge.get_firebase_config()
except ImportError:
    from config_loader import get_config
    firebase_config = {
        "apiKey": get_config("FIREBASE_API_KEY", "AIzaSyDcXAaanNgR2_T0dq2oOl5XyKPksYHppVo"),
        "authDomain": get_config("FIREBASE_AUTH_DOMAIN", "cline-bot.firebaseapp.com"),
        "projectId": get_config("FIREBASE_PROJECT_ID", "cline-bot"),
        "storageBucket": get_config("FIREBASE_STORAGE_BUCKET", "cline-bot.firebasestorage.app"),
        "messagingSenderId": get_config("FIREBASE_MESSAGING_SENDER_ID", "364369702101"),
        "appId": get_config("FIREBASE_APP_ID", "1:364369702101:web:0013885dcf20b43799c65c"),
        "measurementId": get_config("FIREBASE_MEASUREMENT_ID", "G-MDPRELSCD1"),
    }

class FirebaseAuth:
    def __init__(self):
        self.firebase_initialized = False
        self.initialize_firebase()
        
    def initialize_firebase(self):
        """Initialize Firebase if not already done."""
        if not self.firebase_initialized:
            try:
                # Check if Firebase is already initialized
                if not firebase_admin._apps:
                    # Create service account key file
                    service_account_path = self.create_service_account_file()
                    cred = credentials.Certificate(service_account_path)
                    firebase_admin.initialize_app(cred)
                self.firebase_initialized = True
                return True
            except Exception as e:
                st.error(f"Failed to initialize Firebase: {str(e)}")
                return False
        return True
    
    def create_service_account_file(self):
        """Create a service account file from environment variables."""
        # In production, you would get this from environment variables
        # For demo purposes, we'll use a placeholder
        service_account = {
            "type": "service_account",
            "project_id": firebase_config["projectId"],
            # Fill in with actual service account details or 
            # use environment variables in production
            "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID", ""),
            "private_key": os.getenv("FIREBASE_PRIVATE_KEY", "").replace("\\n", "\n"),
            "client_email": os.getenv("FIREBASE_CLIENT_EMAIL", ""),
            "client_id": os.getenv("FIREBASE_CLIENT_ID", ""),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": os.getenv("FIREBASE_CERT_URL", "")
        }
        
        # Create a temp file to store the service account
        temp_dir = Path(os.path.join(os.path.dirname(__file__), "temp"))
        temp_dir.mkdir(exist_ok=True)
        
        service_account_path = temp_dir / "service_account.json"
        with open(service_account_path, "w") as f:
            json.dump(service_account, f)
        
        return service_account_path

    def login_form(self):
        """Display login form and handle authentication."""
        if 'user_info' not in st.session_state:
            st.session_state.user_info = None
        
        # If user is already logged in, show their info
        if st.session_state.user_info:
            st.success(f"Logged in as {st.session_state.user_info['email']}")
            if st.button("Logout"):
                st.session_state.user_info = None
                return False
            return True
        
        # If not logged in, show login form
        with st.form("login_form"):
            st.subheader("Login")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            col1, col2 = st.columns(2)
            login_submitted = col1.form_submit_button("Login")
            demo_mode = col2.form_submit_button("Demo Mode")
            
            if login_submitted:
                if not self.initialize_firebase():
                    st.error("Firebase initialization failed")
                    return False
                
                try:
                    # For demonstration only
                    # In a real implementation, you would use:
                    # user = auth.get_user_by_email(email)
                    # And verify password with Firebase authentication
                    
                    st.session_state.user_info = {"email": email}
                    return True
                except Exception as e:
                    st.error(f"Login failed: {str(e)}")
            
            if demo_mode:
                st.session_state.user_info = {"email": "demo@example.com"}
                return True
        
        # Show registration form in a collapsible section
        with st.expander("New user? Register here"):
            with st.form("register_form"):
                st.subheader("Register")
                new_email = st.text_input("Email Address")
                new_password = st.text_input("Create Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                register_submitted = st.form_submit_button("Register")
                
                if register_submitted:
                    if not self.initialize_firebase():
                        st.error("Firebase initialization failed")
                        return False
                    
                    if new_password != confirm_password:
                        st.error("Passwords do not match")
                        return False
                    
                    try:
                        # For demonstration only
                        # In a real implementation, you would use:
                        # user = auth.create_user(email=new_email, password=new_password)
                        
                        st.success("Registration successful! You can now log in.")
                    except Exception as e:
                        st.error(f"Registration failed: {str(e)}")
        
        return False
