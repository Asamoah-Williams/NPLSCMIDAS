@echo off
REM Navigate to project directory
cd /d "D:\NPL Project\AI Project"
REM Activate virtual environment and run waitress
call .venv\Scripts\activate
waitress-serve --host=127.0.0.1 --port=5001 app:app
