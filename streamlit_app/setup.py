#!/usr/bin/env python
"""
Setup script for Cline Web IDE
This allows easy installation with pip install -e .
"""
from setuptools import setup, find_packages

# Read version from __init__.py
with open("__init__.py", "r") as f:
    exec(f.read())

# Read requirements
with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="cline-web-ide",
    version=__version__,  # Reads from __init__.py
    description="Web-based IDE with AI assistance powered by Streamlit",
    author="Your Name",
    author_email="youremail@example.com",
    url="https://github.com/yourusername/cline-web-ide",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "cline-web=streamlit_app.run:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Web Environment",
        "Framework :: Streamlit",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Integrated Development Environments (IDE)",
    ],
    python_requires=">=3.8",
)
