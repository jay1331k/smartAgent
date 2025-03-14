#!/usr/bin/env python
"""
Quick start script for Cline Web IDE
This script helps new users get started with Cline Web IDE quickly
"""
import os
import sys
import argparse
import subprocess
import webbrowser
from pathlib import Path
import time
import platform

def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")
    try:
        import streamlit
        import anthropic
        print("✓ Core dependencies found")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e.name}")
        return False

def setup_sample_project(target_dir):
    """Set up a sample project for quick testing."""
    print(f"Setting up sample project in {target_dir}...")
    
    # Create project directory structure
    samples_dir = Path(__file__).parent / "samples"
    
    # If samples directory doesn't exist, create it with example files
    if not samples_dir.exists():
        samples_dir.mkdir(exist_ok=True)
        
        # Create a simple Python web app sample
        web_app_dir = samples_dir / "flask_web_app"
        web_app_dir.mkdir(exist_ok=True)
        
        # Create app.py
        with open(web_app_dir / "app.py", "w") as f:
            f.write("""from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', title='Flask Sample App')

@app.route('/about')
def about():
    return render_template('about.html', title='About')

if __name__ == '__main__':
    app.run(debug=True)
""")
        
        # Create requirements.txt
        with open(web_app_dir / "requirements.txt", "w") as f:
            f.write("flask==2.2.3\n")
        
        # Create templates directory
        templates_dir = web_app_dir / "templates"
        templates_dir.mkdir(exist_ok=True)
        
        # Create index.html
        with open(templates_dir / "index.html", "w") as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
        h1 { color: #333; }
        nav { margin-bottom: 20px; }
        nav a { margin-right: 15px; }
    </style>
</head>
<body>
    <nav>
        <a href="/">Home</a>
        <a href="/about">About</a>
    </nav>
    <h1>Welcome to the Sample Flask App</h1>
    <p>This is a simple Flask application created as a sample for Cline Web IDE.</p>
</body>
</html>""")
        
        # Create about.html
        with open(templates_dir / "about.html", "w") as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
        h1 { color: #333; }
        nav { margin-bottom: 20px; }
        nav a { margin-right: 15px; }
    </style>
</head>
<body>
    <nav>
        <a href="/">Home</a>
        <a href="/about">About</a>
    </nav>
    <h1>About This App</h1>
    <p>This is a sample Flask application to demonstrate Cline Web IDE capabilities.</p>
    <p>You can edit and extend this application using Cline's AI assistance.</p>
</body>
</html>""")
        
        print("✓ Created Flask web app sample")
        
        # Create a simple data analysis sample
        data_analysis_dir = samples_dir / "data_analysis"
        data_analysis_dir.mkdir(exist_ok=True)
        
        # Create analysis.py
        with open(data_analysis_dir / "analysis.py", "w") as f:
            f.write("""import pandas as pd
import matplotlib.pyplot as plt

# Load the data
def load_data():
    # You can replace this with your own data or use the sample data
    data = {
        'Year': [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019],
        'Sales': [50000, 60000, 80000, 100000, 120000, 90000, 110000, 130000, 140000, 150000],
        'Expenses': [30000, 35000, 45000, 50000, 55000, 60000, 70000, 80000, 90000, 95000]
    }
    return pd.DataFrame(data)

# Analyze the data
def analyze_data(df):
    # Calculate profit
    df['Profit'] = df['Sales'] - df['Expenses']
    
    # Calculate year-over-year growth
    df['Sales Growth'] = df['Sales'].pct_change() * 100
    df['Profit Growth'] = df['Profit'].pct_change() * 100
    
    return df

# Plot the data
def plot_data(df):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(df['Year'], df['Sales'], marker='o', label='Sales')
    plt.plot(df['Year'], df['Expenses'], marker='s', label='Expenses')
    plt.plot(df['Year'], df['Profit'], marker='^', label='Profit')
    plt.title('Financial Performance')
    plt.xlabel('Year')
    plt.ylabel('Amount ($)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.bar(df['Year'][1:], df['Sales Growth'][1:], label='Sales Growth')
    plt.bar(df['Year'][1:], df['Profit Growth'][1:], alpha=0.7, label='Profit Growth')
    plt.title('Year-over-Year Growth')
    plt.xlabel('Year')
    plt.ylabel('Growth (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('financial_analysis.png')
    plt.show()

# Main function
def main():
    print("Starting financial analysis...")
    df = load_data()
    df = analyze_data(df)
    print(df)
    plot_data(df)
    print("Analysis complete. Check 'financial_analysis.png' for the chart.")

if __name__ == '__main__':
    main()
""")
        
        # Create requirements.txt
        with open(data_analysis_dir / "requirements.txt", "w") as f:
            f.write("pandas==1.5.3\nmatplotlib==3.7.1\n")
            
        print("✓ Created data analysis sample")
    
    # Copy sample to target directory
    import shutil
    
    try:
        if os.path.exists(target_dir):
            # Check if directory is empty
            if os.listdir(target_dir):
                print(f"Warning: Target directory {target_dir} is not empty.")
                response = input("Would you like to continue anyway? Files may be overwritten. (y/n): ")
                if response.lower() != 'y':
                    print("Setup canceled. Please choose an empty directory.")
                    return False
        else:
            os.makedirs(target_dir)
        
        # Copy sample projects
        flask_target = os.path.join(target_dir, "flask_web_app")
        data_analysis_target = os.path.join(target_dir, "data_analysis")
        
        if os.path.exists(samples_dir / "flask_web_app"):
            shutil.copytree(samples_dir / "flask_web_app", flask_target, dirs_exist_ok=True)
        
        if os.path.exists(samples_dir / "data_analysis"):
            shutil.copytree(samples_dir / "data_analysis", data_analysis_target, dirs_exist_ok=True)
        
        print(f"✓ Copied sample projects to {target_dir}")
        return True
    except Exception as e:
        print(f"❌ Error setting up sample project: {str(e)}")
        return False

def launch_ide(port, no_auth):
    """Launch the Cline Web IDE."""
    print("Launching Cline Web IDE...")
    
    # Start the IDE in a new process
    cmd = [sys.executable, "run.py"]
    
    if port:
        cmd.extend(["--port", str(port)])
    
    if no_auth:
        cmd.append("--no-auth")
    
    # On Windows, use Popen to avoid blocking 
    # On Unix, simply open a new terminal
    try:
        if platform.system() == "Windows":
            subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:
            # On macOS and Linux, try to open a new terminal window
            if platform.system() == "Darwin":  # macOS
                os.system(f"osascript -e 'tell app \"Terminal\" to do script \"cd {os.getcwd()} && {' '.join(cmd)}\"'")
            else:  # Linux
                # Try different terminal emulators
                terminals = ["gnome-terminal", "konsole", "xterm"]
                launched = False
                for terminal in terminals:
                    try:
                        subprocess.call([terminal, "--", "bash", "-c", f"cd {os.getcwd()} && {' '.join(cmd)}"])
                        launched = True
                        break
                    except FileNotFoundError:
                        continue
                
                if not launched:
                    # Fallback to running in the same terminal
                    subprocess.Popen(cmd)
        
        # Wait a moment for the server to start
        print("Starting server...")
        time.sleep(3)
        
        # Open browser
        url = f"http://localhost:{port or 8501}"
        print(f"Opening {url} in your web browser...")
        webbrowser.open(url)
        
        print("""
╔════════════════════════════════════════════════════╗
║                 Cline Web IDE                      ║
╠════════════════════════════════════════════════════╣
║ The IDE is now running in your browser.            ║
║ Close the terminal window to stop the server.      ║
║                                                    ║
║ You can access the IDE at:                         ║
║ http://localhost:{port}                            ║
║                                                    ║
║ For help and documentation, visit:                 ║
║ https://github.com/yourusername/cline-web-ide      ║
╚════════════════════════════════════════════════════╝
""".format(port=port or 8501))
        
        return True
    except Exception as e:
        print(f"❌ Error launching IDE: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick start for Cline Web IDE")
    parser.add_argument("--sample", action="store_true", help="Set up a sample project")
    parser.add_argument("--dir", type=str, default="./sample_project", help="Directory for sample project")
    parser.add_argument("--port", type=int, default=8501, help="Port for the IDE server")
    parser.add_argument("--no-auth", action="store_true", help="Disable authentication")
    
    args = parser.parse_args()
    
    if not check_dependencies():
        print("Installing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    if args.sample:
        setup_sample_project(args.dir)
    
    launch_ide(args.port, args.no_auth)
