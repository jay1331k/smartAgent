import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Try to import yaml, but handle case where it's not installed
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

def load_env_variables():
    """Load environment variables from .env file."""
    # Try to find .env file in different locations
    env_paths = [
        Path.cwd() / '.env',
        Path.cwd().parent / '.env',
        Path.home() / '.cline-web-ide' / '.env'
    ]
    
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            return True
    
    # If no .env file found, load from default environment
    load_dotenv()
    return False

def get_anthropic_api_key():
    """Get Anthropic API key from environment variables."""
    # Try different possible environment variable names
    for var_name in ['ANTHROPIC_API_KEY', 'CLAUDE_API_KEY', 'CLINE_API_KEY']:
        api_key = os.environ.get(var_name)
        if api_key:
            return api_key
    return None

def get_firebase_service_account():
    """Get Firebase service account from environment variables or file."""
    # Try to get from environment variable first
    service_account_json = os.environ.get('FIREBASE_SERVICE_ACCOUNT')
    if service_account_json:
        try:
            return json.loads(service_account_json)
        except json.JSONDecodeError:
            pass
    
    # Try to get from file
    service_account_paths = [
        Path.cwd() / 'firebase-service-account.json',
        Path.cwd().parent / 'firebase-service-account.json',
        Path.home() / '.cline-web-ide' / 'firebase-service-account.json'
    ]
    
    for path in service_account_paths:
        if path.exists():
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except:
                pass
    
    return None

def get_config(config_name, default=None):
    """Get configuration value from environment variables or config files."""
    # Try environment variable first
    env_var = os.environ.get(f'CLINE_WEB_IDE_{config_name.upper()}')
    if env_var:
        return env_var
    
    # Try config files
    config_paths = [
        Path.cwd() / 'config.json',
        Path.cwd() / 'config.yaml',
        Path.cwd().parent / 'config.json',
        Path.cwd().parent / 'config.yaml',
        Path.home() / '.cline-web-ide' / 'config.json',
        Path.home() / '.cline-web-ide' / 'config.yaml'
    ]
    
    for config_path in config_paths:
        if config_path.exists():
            try:
                if config_path.suffix == '.json':
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                elif config_path.suffix == '.yaml' and YAML_AVAILABLE:
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                else:
                    # Skip yaml files if yaml module is not available
                    continue
                
                if config_name in config:
                    return config[config_name]
            except:
                pass
    
    return default
