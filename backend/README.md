# ForensicAI Backend API

This is the backend API for the ForensicAI application, which provides video analysis and forensic capabilities.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

1. Clone the repository
2. Navigate to the backend directory
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file from the template:

```bash
cp .env.example .env
# Edit the .env file with your own values
```

### Running the Server

#### Development Mode

```bash
# Using the helper script (recommended)
python run.py

# Or directly with Flask
python server.py
```

#### Production Mode

```bash
# Using the helper script with gunicorn
python run.py --prod

# Or directly with gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 server:app
```

## API Documentation

### Status Endpoint

**GET /api/status**

Returns the status of the API.

```json
{
  "status": "online",
  "version": "1.0.0",
  "message": "Forensic AI API is running",
  "backend_modules": true
}
```

### Frames Endpoints

**GET /api/frames**

Returns a list of all frames.

```json
{
  "frames": [
    {
      "id": "frame_001",
      "timestamp": "2024-03-24 10:45:23",
      "thumbnail": "https://placehold.co/400x225",
      "detections": ["Person", "Laptop", "Chair"],
      "confidence": 0.95
    },
    ...
  ],
  "count": 3
}
```

**GET /api/frames/:frameId**

Returns a specific frame by ID.

```json
{
  "id": "frame_001",
  "timestamp": "2024-03-24 10:45:23",
  "thumbnail": "https://placehold.co/400x225",
  "detections": ["Person", "Laptop", "Chair"],
  "confidence": 0.95
}
```

### Ask Endpoint

**POST /api/ask**

Ask a question about a specific frame.

Request body:

```json
{
  "question": "What objects are in this frame?",
  "frameId": "frame_001"
}
```

Response:

```json
{
  "answer": "In this frame, I can see a person sitting at a desk with a laptop and chair.",
  "confidence": 0.92,
  "frameId": "frame_001",
  "timestamp": "2024-04-10 14:23:01"
}
```

### Proxy Endpoint

**ANY /proxy/:url**

Forwards requests to the specified URL to avoid CORS issues.

## Error Handling

All endpoints return appropriate HTTP status codes and error messages in the following format:

```json
{
  "error": "Description of the error"
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
