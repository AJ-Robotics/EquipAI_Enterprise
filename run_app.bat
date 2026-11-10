@echo off
TITLE EquipAI Enterprise Launcher
COLOR 0B

echo ==========================================
echo    EquipAI Chatbot Launcher (CPU-Optimized)
echo ==========================================
echo.
cd /d "%~dp0"

echo [1/2] Validating environment...

:: Verify python is accessible
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [!] Python is not installed or not in PATH! Please install Python.
    pause
    exit /b 1
)

:: Verify Streamlit is installed, otherwise install requirements
python -c "import streamlit" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [!] Streamlit not found. Installing dependencies...
    pip install -r requirements.txt
)

echo [2/2] Launching Application...
echo.

:: -------- CPU tuning knobs (overridable, safe defaults) --------
:: Force CPU inference even if a GPU exists but is weak.
set EQPAI_USE_GPU=0
:: Hybrid retrieval = fault-code exact match + keyword + vector search.
set EQPAI_HYBRID=1
:: Leave some cores free for the UI / other apps on normal PCs.
if "%EQPAI_LLM_THREADS%"=="" set EQPAI_LLM_THREADS=4
if "%EQPAI_EMBED_THREADS%"=="" set EQPAI_EMBED_THREADS=2
:: Generation temperature 0 disables repeat_penalty on some llama.cpp builds.
if "%EQPAI_TEMPERATURE%"=="" set EQPAI_TEMPERATURE=0.2
:: Default response mode: fast / balanced / research
if "%EQPAI_DEFAULT_MODE%"=="" set EQPAI_DEFAULT_MODE=balanced

echo ------------------------------------------
echo    App will open in your browser shortly.
echo    Keep this window open to keep the chat active.
echo ------------------------------------------
echo.

:: Explicitly use python -m streamlit to prevent code editors from intercepting
python -m streamlit run app.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] Application crashed or failed to start.
    pause
)