#!/bin/bash
cd "$(dirname "$0")"
APP_DIR="api"
VENV_PYTHON="$APP_DIR/venv/bin/python3"
LOG_DIR="logs"
mkdir -p $LOG_DIR

# 1. Ensure Dependencies
if [ ! -d "$APP_DIR/venv" ]; then
    echo "Setting up virtual environment..."
    python3 -m venv $APP_DIR/venv
    $APP_DIR/venv/bin/pip install --upgrade pip
    $APP_DIR/venv/bin/pip install Flask Flask-Cors numpy google-genai
fi

# 2. Check & Start Backend (Port 5001)
if ! lsof -i:5001 > /dev/null; then
    echo "Starting Backend on 5001..."
    export GEMINI_API_KEY="AIzaSyBLcCZ0u2PhNlo4zPkNI5QMM8V9REXDI7w"
    nohup $VENV_PYTHON $APP_DIR/app.py > $LOG_DIR/app_startup.log 2>&1 &
    sleep 2
else
    echo "Backend is already running."
fi

# 3. Check & Start UI Proxy (Port 5007)
if ! lsof -i:5007 > /dev/null; then
    echo "Starting UI Proxy on 5007..."
    nohup socat OPENSSL-LISTEN:5007,fork,reuseaddr,cert=/etc/letsencrypt/live/DOMAIN/fullchain.pem,key=/etc/letsencrypt/live/DOMAIN/privkey.pem,verify=0 TCP:127.0.0.1:5001 > $LOG_DIR/socat_5007.log 2>&1 &
else
    echo "UI Proxy is already running."
fi

# 4. Check & Start API Proxy (Port 5012)
if ! lsof -i:5012 > /dev/null; then
    echo "Starting API Proxy on 5012..."
    nohup socat OPENSSL-LISTEN:5012,fork,reuseaddr,cert=/etc/letsencrypt/live/DOMAIN/fullchain.pem,key=/etc/letsencrypt/live/DOMAIN/privkey.pem,verify=0 TCP:127.0.0.1:5001 > $LOG_DIR/socat_5012.log 2>&1 &
else
    echo "API Proxy is already running."
fi

echo "Health check complete."
