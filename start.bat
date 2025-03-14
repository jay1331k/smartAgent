@echo off
REM filepath: e:\Projects\smartAgent\start.bat
echo Starting Smart Agent IDE...

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python not found. Please install Python and try again.
    pause
    exit /b 1
)

REM Check if requirements are installed
echo Checking dependencies...
python -c "import streamlit; import google.generativeai" >nul 2>&1
if %errorlevel% neq 0 (
    echo Some dependencies are missing. Would you like to install them? (Y/N)
    set /p INSTALL=
    if /i "%INSTALL%"=="Y" (
        echo Installing dependencies...
        pip install -r requirements.txt
        if %errorlevel% neq 0 (
            echo Failed to install dependencies. Please run 'pip install -r requirements.txt' manually.
            pause
            exit /b 1
        )
    ) else (
        echo Skipping dependency installation. The application may not work correctly.
    )
)

REM Run the application
echo Starting app...
python run.py
if %errorlevel% neq 0 (
    echo The application encountered an error. Check the output above for details.
    pause
)
