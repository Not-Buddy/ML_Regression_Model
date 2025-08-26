#!/bin/bash

# Activate Python virtual environment script
# Usage: ./activateenv.sh

VENV_PATH="myenv"  # Change this to your virtual environment path

echo "🔍 Looking for virtual environment at: $VENV_PATH"

if [ -d "$VENV_PATH" ]; then
    if [ -f "$VENV_PATH/bin/activate" ]; then
        echo "🚀 Activating virtual environment..."
        source "$VENV_PATH/bin/activate"
        echo "✅ Virtual environment activated!"
        echo "📍 Python location: $(which python)"
        echo "🐍 Python version: $(python --version)"
    else
        echo "❌ Activate script not found at $VENV_PATH/bin/activate"
        exit 1
    fi
else
    echo "❌ Virtual environment directory not found: $VENV_PATH"
    echo "💡 Create it with: python -m venv $VENV_PATH"
    exit 1
fi

