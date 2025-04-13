# ForensicAI

ForensicAI is a comprehensive video analysis system that combines object detection and forensic intelligence powered by large language models (LLMs).

## Features

- **Real-time Video Analysis**: Process live camera feeds or existing video files
- **Object Detection**: Identify and track objects using YOLO
- **LLM-Powered Insights**: Generate descriptions and insights for detected scenes
- **Forensic Timeline**: Visualize events in a chronological timeline
- **Interactive Dashboard**: Modern React frontend with real-time updates
- **Frame Analysis**: Ask questions about specific video frames

## Architecture

The project consists of two main components:

1. **Backend**: Python-based server using Flask for API endpoints and video processing
2. **Frontend**: React application with Tailwind CSS for the user interface

## Installation

### Prerequisites

- Python 3.8 or higher
- Node.js 16 or higher
- npm or yarn

### Automated Installation

Use the provided installation script:

```bash
./install.sh
```

### Manual Installation

#### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your configuration
```

#### Frontend

```bash
cd frontend
npm install
```

## Running the Application

### Development Mode

```bash
# From the frontend directory
npm run start
```

This will start both the backend Flask server and the frontend Vite development server.

### Production Mode

Backend:

```bash
cd backend
source venv/bin/activate
python run.py --prod
```

Frontend:

```bash
cd frontend
npm run build
```

The built frontend will be served by the Flask server at http://localhost:5000

## API Documentation

See [backend/README.md](backend/README.md) for detailed API documentation.

## License

MIT License
