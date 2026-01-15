@echo off
title EcoDrone AI - Controller
color 0A
echo ========================================================
echo      EcoDrone AI - Afforestation Monitoring System
echo      Built by Soumoditya Das for Kshitij 2026
echo ========================================================

echo.
echo [1/4] Checking Environment...
python --version
node --version
echo.

echo [2/4] Installing Backend Services...
cd backend
pip install -r requirements.txt
pip install Pillow
if %errorlevel% neq 0 (
    echo [!] Warning: Some dependencies failed. Check Internet.
)
cd ..

echo [3/4] Ensuring Sample Data Exists...
if not exist "sample_op1.png" (
    echo [+] Generating Synthetic Drone Imagery...
    python generate_samples.py
)

echo [4/4] Launching Services...
echo     - Starting FastAPI Backend on Port 8000
start "EcoDrone Backend" cmd /k "cd backend && python -m uvicorn app.main:app --reload --port 8000"

echo     - Starting React Frontend on Port 5173
cd frontend
call npm install
start "EcoDrone Frontend" cmd /k "npm run dev"

echo.
echo ========================================================
echo    SYSTEM ONLINE
echo    Frontend: http://localhost:5173
echo    Backend:  http://localhost:8000/docs
echo ========================================================
echo.
pause
