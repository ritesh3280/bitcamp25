#!/bin/bash
# Installation script for ForensicAI

echo "Installing ForensicAI..."

# Create virtual environment for backend
echo "Setting up Python virtual environment..."
cd backend
python -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Copy .env file
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please edit the .env file with your own values."
fi

# Deactivate virtual environment
deactivate
cd ..

# Install frontend dependencies
echo "Installing frontend dependencies..."
cd frontend
npm install

echo "Installation complete!"
echo "To start the development server, run: cd frontend && npm run start" 