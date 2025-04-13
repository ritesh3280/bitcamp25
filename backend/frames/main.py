#!/usr/bin/env python3
"""
Video Analysis System - Main Entry Point

This script provides a unified interface to:
1. Record and analyze live camera/screen feed
2. Process existing video files

Both modes use the same object detection and LLM description pipeline.
"""

import os
import sys
import argparse
import cv2
import time
import json
import numpy as np
from datetime import datetime
import threading
from flask import Flask, jsonify, request, Response, send_file
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
from ultralytics import YOLO
from functools import wraps
import uuid
import logging
import tempfile
import concurrent.futures
import shutil

# Try to import the video processing module
try:
    from main4 import process_video_file
    video_processor_available = True
except ImportError:
    print("Video processing module (main4.py) not available")
    video_processor_available = False

# Import only the functions we need from main4
from main4 import (
    initialize_camera, 
    load_yolo_model, 
    process_video_file
)
# Import the rest of main4 as a module for accessing classes
import main4

# Import our Pinecone utilities
from pinecone_utils import upsert_session_vectors, semantic_search_frames

# Global variables for frame management
frame_lock = threading.Lock()
latest_frame = None
latest_detections = []
analysis_status = {
    "is_running": False,
    "frame_count": 0,
    "unique_frames": 0,
    "session_dir": "",
    "start_time": None,
    "fps": 0
}

# Global variables for live feed control
live_feed_thread = None
stop_live_feed = threading.Event()
camera_id = 1
max_api_calls = 10
threshold = 30

# Create Flask app
app = Flask(__name__)

# Standard CORS setup that should work for all cases
CORS(app, resources={r"/*": {
    "origins": "*",  # Allow all origins
    "methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"]
}}, supports_credentials=False)  # Changed to False for standard CORS with any origin

@app.route('/api/status', methods=['GET'])
def get_status():
    """Return the current status of the live analysis"""
    with frame_lock:
        status_copy = analysis_status.copy()
        if status_copy["start_time"]:
            status_copy["elapsed_time"] = time.time() - status_copy["start_time"]
        else:
            status_copy["elapsed_time"] = 0
        
        # Include count of current detections
        status_copy["current_detections"] = len(latest_detections)
        
        # Don't include the start_time in the response (not JSON serializable)
        del status_copy["start_time"]
        
        return jsonify(status_copy)

@app.route('/api/latest-frame', methods=['GET'])
def get_latest_frame():
    """Return the latest frame as a JPEG image"""
    with frame_lock:
        if latest_frame is not None:
            try:
                # Print debug info
                print(f"Sending frame: {latest_frame.shape}, type: {type(latest_frame)}")
                
                # Encode the frame as JPEG - adjust quality if needed
                _, buffer = cv2.imencode('.jpg', latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                
                # Create response with correct headers
                response = Response(buffer.tobytes(), mimetype='image/jpeg')
                response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
                response.headers['Pragma'] = 'no-cache'
                response.headers['Expires'] = '0'
                
                return response
            except Exception as e:
                print(f"Error sending frame: {e}")
                return Response(status=500)
        else:
            print("No frame available yet")
            return Response(status=404)

@app.route('/api/latest-detections', methods=['GET'])
def get_latest_detections():
    """Return the latest detections"""
    with frame_lock:
        return jsonify(latest_detections)

@app.route('/api/control', methods=['POST'])
def control_analysis():
    """Control the analysis (pause/resume/start/stop)"""
    global analysis_status, live_feed_thread, stop_live_feed, camera_id, max_api_calls, threshold
    
    data = request.json
    if not data:
        return jsonify({"error": "Invalid request"}), 400
    
    command = data.get('command')
    
    if command == 'pause':
        # Logic to pause analysis
        analysis_status["is_running"] = False
        return jsonify({"status": "paused"})
    
    elif command == 'resume':
        # Logic to resume analysis
        analysis_status["is_running"] = True
        return jsonify({"status": "running"})
    
    elif command == 'start_live':
        # Check if live feed is already running
        if live_feed_thread and live_feed_thread.is_alive():
            return jsonify({"status": "live_feed_already_running"})
        
        # Get parameters from request if provided
        if 'camera_id' in data:
            camera_id = data.get('camera_id')
        if 'max_api_calls' in data:
            max_api_calls = data.get('max_api_calls')
        if 'threshold' in data:
            threshold = data.get('threshold')
        
        # Reset stop event
        stop_live_feed.clear()
        
        # Start live feed in a separate thread
        live_feed_thread = threading.Thread(
            target=run_live_recording_thread,
            args=(camera_id, max_api_calls, threshold)
        )
        live_feed_thread.daemon = True
        live_feed_thread.start()
        
        return jsonify({"status": "live_feed_started", "camera_id": camera_id})
    
    elif command == 'stop_live':
        # Check if live feed is running
        if not live_feed_thread or not live_feed_thread.is_alive():
            return jsonify({"status": "live_feed_not_running"})
        
        # Signal thread to stop and reset status
        stop_live_feed.set()
        with frame_lock:
            analysis_status["is_running"] = False
        
        # Wait for thread to finish (with timeout)
        live_feed_thread.join(timeout=3.0)
        
        # Release OpenCV windows
        cv2.destroyAllWindows()
        
        # Clean up status
        with frame_lock:
            global latest_frame
            latest_frame = None  # Clear the latest frame to avoid showing stale images
            latest_detections.clear()
            analysis_status["frame_count"] = 0
            analysis_status["unique_frames"] = 0
            analysis_status["elapsed_time"] = 0
            analysis_status["start_time"] = None
        
        return jsonify({"status": "live_feed_stopped"})
    
    else:
        return jsonify({"error": f"Unknown command: {command}"}), 400

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    """Return a list of all available video insights sessions"""
    try:
        base_dir = "video_insights"
        if not os.path.exists(base_dir):
            return jsonify({"error": "No sessions found"}), 404
        
        # Get list of all session directories
        sessions = []
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                # Check if keyframes.json exists
                keyframes_path = os.path.join(item_path, "keyframes.json")
                if os.path.exists(keyframes_path):
                    # Get creation time from metadata or use file creation time
                    try:
                        with open(keyframes_path, 'r') as f:
                            metadata = json.load(f).get('metadata', {})
                            created = metadata.get('created', None)
                    except:
                        created = None
                    
                    if not created:
                        # Fallback to directory creation time
                        created = datetime.fromtimestamp(os.path.getctime(item_path)).isoformat()
                    
                    # Get frame count
                    frame_count = 0
                    images_dir = os.path.join(item_path, "images")
                    if os.path.exists(images_dir):
                        frame_count = len([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
                    
                    sessions.append({
                        "id": item,
                        "name": item.replace("_", " ").title(),
                        "created": created,
                        "frame_count": frame_count,
                        "path": item_path
                    })
        
        # Sort by creation time (newest first)
        sessions.sort(key=lambda x: x['created'], reverse=True)
        
        return jsonify({"sessions": sessions})
    
    except Exception as e:
        print(f"Error getting sessions: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/sessions/<session_id>', methods=['GET'])
def get_session_details(session_id):
    """Return detailed information about a specific session"""
    try:
        base_dir = "video_insights"
        session_dir = os.path.join(base_dir, session_id)
        
        if not os.path.exists(session_dir):
            return jsonify({"error": "Session not found"}), 404
        
        keyframes_path = os.path.join(session_dir, "keyframes.json")
        if not os.path.exists(keyframes_path):
            return jsonify({"error": "Session data not found"}), 404
        
        with open(keyframes_path, 'r') as f:
            session_data = json.load(f)
        
        # Extract metadata
        metadata = session_data.get('metadata', {})
        
        # Count frames
        frame_count = len(session_data.get('frames', []))
        
        return jsonify({
            "id": session_id,
            "name": session_id.replace("_", " ").title(),
            "created": metadata.get('created', ''),
            "frame_count": frame_count,
            "metadata": metadata
        })
    
    except Exception as e:
        print(f"Error getting session details: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/sessions/<session_id>/frames', methods=['GET'])
def get_session_frames(session_id):
    """Return frames from a specific session"""
    try:
        base_dir = "video_insights"
        session_dir = os.path.join(base_dir, session_id)
        
        if not os.path.exists(session_dir):
            return jsonify({"error": "Session not found"}), 404
        
        keyframes_path = os.path.join(session_dir, "keyframes.json")
        if not os.path.exists(keyframes_path):
            return jsonify({"error": "Session data not found"}), 404
        
        with open(keyframes_path, 'r') as f:
            session_data = json.load(f)
        
        frames = session_data.get('frames', [])
        
        # Add debug logging
        print(f"DEBUG: First frame keys in keyframes.json: {list(frames[0].keys()) if frames else 'No frames'}")
        if frames and 'llm_description' in frames[0]:
            print(f"DEBUG: llm_description exists in first frame: {frames[0]['llm_description']}")
        
        # Create a simplified view of frames for the frontend
        simplified_frames = []
        for frame in frames:
            frame_data = {
                "id": frame.get('frame_id', ''),
                "timestamp": frame.get('timestamp', ''),
                "elapsed_time": frame.get('elapsed_time', 0),
                "image_path": frame.get('image_path', ''),
                "objects": frame.get('objects', []),
                "confidence": max(frame.get('confidence', {}).values()) if frame.get('confidence') else 0,
                # Include the llm_description if it exists
                "llm_description": frame.get('llm_description', None)
            }
            simplified_frames.append(frame_data)
        
        # Add debug logging for response
        if simplified_frames:
            print(f"DEBUG: First simplified frame keys: {list(simplified_frames[0].keys())}")
            if 'llm_description' in simplified_frames[0]:
                print(f"DEBUG: llm_description value in simplified frame: {simplified_frames[0]['llm_description']}")
            else:
                print("DEBUG: llm_description missing from simplified frame")
        
        # Asynchronously upload frame vectors to Pinecone
        # We do this in a separate thread to avoid blocking the response
        def upload_vectors_in_background():
            try:
                upsert_session_vectors(session_id, frames)
            except Exception as e:
                print(f"Error upserting vectors for session {session_id}: {e}")
        
        # Start a background thread to upsert vectors without blocking response
        upload_thread = threading.Thread(target=upload_vectors_in_background)
        upload_thread.daemon = True
        upload_thread.start()
        
        return jsonify({"frames": simplified_frames})
    
    except Exception as e:
        print(f"Error getting session frames: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/frames/<frame_id>/image', methods=['GET'])
def get_frame_image(frame_id):
    """Return the image for a specific frame"""
    try:
        # Search for the frame in all sessions
        base_dir = "video_insights"
        frame_path = None
        
        for session in os.listdir(base_dir):
            session_dir = os.path.join(base_dir, session)
            if not os.path.isdir(session_dir):
                continue
                
            images_dir = os.path.join(session_dir, "images")
            if not os.path.exists(images_dir):
                continue
                
            potential_path = os.path.join(images_dir, f"{frame_id}.jpg")
            if os.path.exists(potential_path):
                frame_path = potential_path
                break
        
        if not frame_path:
            return jsonify({"error": "Frame not found"}), 404
        
        # Return the image
        return send_file(frame_path, mimetype='image/jpeg')
    
    except Exception as e:
        print(f"Error getting frame image: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    """Simplified chat endpoint for frame analysis that avoids CORS issues"""
    # No need for manual CORS handling - Flask-CORS extension takes care of it
    if request.method == 'OPTIONS':
        return app.make_default_options_response()
        
    try:
        # For actual POST requests
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        frame_id = data.get('frame_id')
        question = data.get('question')
        conversation_history = data.get('conversation_history', [])
        
        if not frame_id or not question:
            return jsonify({"error": "Missing frame_id or question"}), 400
            
        # Search for the frame
        base_dir = "video_insights"
        frame_path = None
        frame_data = None
        
        # Find frame in any session
        for session in os.listdir(base_dir):
            session_dir = os.path.join(base_dir, session)
            if not os.path.isdir(session_dir):
                continue
                
            images_dir = os.path.join(session_dir, "images")
            if not os.path.exists(images_dir):
                continue
                
            potential_path = os.path.join(images_dir, f"{frame_id}.jpg")
            if os.path.exists(potential_path):
                frame_path = potential_path
                
                # Get frame metadata
                keyframes_path = os.path.join(session_dir, "keyframes.json")
                if os.path.exists(keyframes_path):
                    with open(keyframes_path, 'r') as f:
                        keyframes_data = json.load(f)
                        for frame in keyframes_data.get('frames', []):
                            if frame.get('frame_id') == frame_id:
                                frame_data = frame
                                break
                break
                
        if not frame_path:
            return jsonify({"error": "Frame not found"}), 404
            
        # Get objects from frame data
        objects_text = "No objects detected"
        if frame_data and frame_data.get('objects'):
            objects_text = ", ".join(frame_data.get('objects'))
            
        # Import OpenAI
        client = OpenAI()
        
        # Prepare conversation for API
        messages = [
            {"role": "system", "content": (
                "You are an AI assistant skilled in analyzing video frames. "
                "You'll be given information about objects detected in a frame and sometimes "
                "asked questions about what's visible. Answer concisely and accurately based "
                "on what's detectable in the frame. If you can't determine something from "
                "the available information, acknowledge the limitation rather than speculating."
            )}
        ]
        
        # Add conversation history if provided
        if conversation_history:
            for msg in conversation_history:
                messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Add frame image as base64
        import base64
        with open(frame_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
        # Add image context
        frame_context = f"This frame contains the following detected objects: {objects_text}."
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": frame_context},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                    }
                }
            ]
        })
        
        # Add user question
        messages.append({"role": "user", "content": question})
        
        # Call GPT-4o
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1000
        )
        
        answer = completion.choices[0].message.content
        
        # Prepare response
        result = {
            "answer": answer,
            "frame_id": frame_id,
            "timestamp": datetime.now().isoformat(),
            "conversation": [
                *conversation_history,
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/cameras/check', methods=['POST'])
def check_camera():
    """Check if a camera is available and can be accessed"""
    data = request.json
    if not data:
        return jsonify({"error": "Invalid request"}), 400

    camera_id = data.get('camera_id', 0)
    
    try:
        # Try to initialize the camera
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            return jsonify({
                "available": False,
                "error": f"Camera {camera_id} could not be opened"
            })
        
        # Try to read a frame to make sure it works
        ret, frame = cap.read()
        
        if not ret or frame is None:
            cap.release()
            return jsonify({
                "available": False,
                "error": f"Camera {camera_id} could not capture frames"
            })
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Release the camera
        cap.release()
        
        return jsonify({
            "available": True,
            "camera_id": camera_id,
            "resolution": f"{width}x{height}",
            "fps": fps
        })
        
    except Exception as e:
        return jsonify({
            "available": False,
            "error": f"Error checking camera: {str(e)}"
        })

@app.route('/api/ping', methods=['GET', 'OPTIONS'])
def ping():
    """Simple endpoint to test CORS and connectivity"""
    if request.method == 'OPTIONS':
        return app.make_default_options_response()
        
    return jsonify({"status": "ok", "message": "Backend API is up and running"})

@app.route('/api/frames/semantic-search', methods=['POST', 'OPTIONS'])
def semantic_search_frames_endpoint():
    """Endpoint for semantic search across frames in a session using Pinecone vector database"""
    # No need for manual CORS handling - Flask-CORS extension takes care of it
    if request.method == 'OPTIONS':
        return app.make_default_options_response()
        
    try:
        # For actual POST requests
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        session_id = data.get('session_id')
        query = data.get('query')
        
        if not session_id or not query:
            return jsonify({"error": "Missing session_id or query"}), 400
            
        # Find the session directory
        base_dir = "video_insights"
        session_dir = os.path.join(base_dir, session_id)
        
        if not os.path.isdir(session_dir):
            return jsonify({"error": f"Session {session_id} not found"}), 404
        
        # Use Pinecone search
        try:
            search_results = semantic_search_frames(session_id, query, top_k=5)
            
            # Format the response
            response_data = {
                "results": search_results
            }
            
            return jsonify(response_data)
            
        except Exception as e:
            # If Pinecone search fails, we'll have to upsert vectors first
            print(f"Pinecone search failed: {e}")
            print("This may be because vectors need to be upserted first. Trying to upsert now...")
            
            # Load keyframes data
            keyframes_path = os.path.join(session_dir, "keyframes.json")
            if not os.path.exists(keyframes_path):
                return jsonify({"error": f"Keyframes data not found for session {session_id}"}), 404
                
            with open(keyframes_path, 'r') as f:
                keyframes_data = json.load(f)
                
            # Extract frames
            frames = keyframes_data.get('frames', [])
            
            # Upsert vectors
            success = upsert_session_vectors(session_id, frames)
            
            if not success:
                return jsonify({"error": "Failed to vectorize frames for search"}), 500
                
            # Try search again
            try:
                search_results = semantic_search_frames(session_id, query, top_k=5)
                
                # Format the response
                response_data = {
                    "results": search_results
                }
                
                return jsonify(response_data)
            except Exception as e2:
                print(f"Second attempt at Pinecone search failed: {e2}")
                return jsonify({"error": f"Failed to perform search after vectorization: {str(e2)}"}), 500
            
    except Exception as e:
        print(f"Error in semantic search endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/api/upload', methods=['POST', 'OPTIONS'])
def upload_file_endpoint():
    """
    Upload a video file to the server
    """
    # No need to handle CORS manually - Flask-CORS extension takes care of it
    if request.method == 'OPTIONS':
        return app.make_default_options_response()
    
    try:
        # Check if file was sent
        if 'video' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['video']
        
        # Check if filename is empty
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Create uploads directory if it doesn't exist
        uploads_dir = os.path.join(os.path.dirname(__file__), 'uploads')
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)
        
        # Save the file
        file_path = os.path.join(uploads_dir, file.filename)
        file.save(file_path)
        
        return jsonify({
            "success": True,
            "message": "File uploaded successfully",
            "filename": file.filename
        })
        
    except Exception as e:
        print(f"Error uploading file: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/process-video', methods=['POST', 'OPTIONS'])
def process_video_endpoint():
    """
    Process a video file with the video_insights module
    """
    # No need to handle CORS manually - Flask-CORS extension takes care of it
    if request.method == 'OPTIONS':
        return app.make_default_options_response()
    
    # Handle the actual request
    try:
        # Parse JSON data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Extract parameters
        filename = data.get('filename')
        frame_interval = data.get('frame_interval', 15)
        max_api_calls = data.get('max_api_calls', 10)
        threshold = data.get('threshold', 30)
        mock_file = data.get('mock_file', False)  # Check if this is a mock request
        
        if not filename:
            return jsonify({"error": "Filename is required"}), 400
        
        # Check if video processing is available
        if not video_processor_available:
            return jsonify({"error": "Video processing module is not available"}), 500
        
        # Check if the video file exists in the uploads directory
        uploads_dir = os.path.join(os.path.dirname(__file__), 'uploads')
        video_path = os.path.join(uploads_dir, filename)
        
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)
            
        # If the file doesn't exist and mock_file is true, use a default sample video
        # or create a mock file entry in the sessions.json
        if not os.path.exists(video_path):
            if mock_file:
                print(f"Mock file requested for {filename}")
                # Generate a session ID
                session_id = str(uuid.uuid4())[:8]
                
                # For demo, we'll pretend processing happened without actually doing it
                # Add session directly to sessions.json
                sessions_file = os.path.join(os.path.dirname(__file__), 'sessions.json')
                try:
                    if os.path.exists(sessions_file):
                        with open(sessions_file, 'r') as f:
                            sessions = json.load(f)
                    else:
                        sessions = {"sessions": []}
                    
                    # Add new mock session
                    sessions["sessions"].append({
                        "id": session_id,
                        "name": f"Analysis of {filename} (Mock)",
                        "created": datetime.now().isoformat(),
                        "video_path": video_path,
                        "frame_count": 0,
                        "is_mock": True
                    })
                    
                    # Save back to the file
                    with open(sessions_file, 'w') as f:
                        json.dump(sessions, f, indent=2)
                    
                    # Return success response for mock processing
                    return jsonify({
                        "success": True,
                        "message": "Mock video processing started",
                        "session_id": session_id
                    })
                    
                except Exception as e:
                    print(f"Error creating mock session: {e}")
                    return jsonify({"error": str(e)}), 500
            else:
                return jsonify({"error": f"Video file {filename} not found in uploads directory"}), 404
        
        # Generate a session ID
        session_id = str(uuid.uuid4())[:8]
        
        # Process the video in a background thread to avoid blocking
        def process_video_thread():
            try:
                # Call the video processing function
                result_dir = process_video_file(
                    video_path=video_path,
                    frame_interval=frame_interval,
                    max_api_calls=max_api_calls,
                    threshold=threshold
                )
                
                print(f"Video processing complete. Results in: {result_dir}")
                
                # Update session info with the result directory for later retrieval
                # This would normally be stored in a database
                sessions_file = os.path.join(os.path.dirname(__file__), 'sessions.json')
                try:
                    if os.path.exists(sessions_file):
                        with open(sessions_file, 'r') as f:
                            sessions = json.load(f)
                    else:
                        sessions = {"sessions": []}
                    
                    # Add new session
                    sessions["sessions"].append({
                        "id": session_id,
                        "name": f"Analysis of {filename}",
                        "created": datetime.now().isoformat(),
                        "video_path": video_path,
                        "result_dir": result_dir,
                        "frame_count": len(os.listdir(os.path.join(result_dir, "images"))) if result_dir else 0
                    })
                    
                    # Save back to the file
                    with open(sessions_file, 'w') as f:
                        json.dump(sessions, f, indent=2)
                        
                except Exception as e:
                    print(f"Error updating sessions file: {e}")
                    
            except Exception as e:
                print(f"Error in video processing thread: {e}")
        
        # Start the processing thread
        threading.Thread(target=process_video_thread).start()
        
        # Return success with the session ID
        return jsonify({
            "success": True,
            "message": "Video processing started",
            "session_id": session_id
        })
        
    except Exception as e:
        print(f"Error processing video: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/process-video-test', methods=['POST', 'OPTIONS'])
def process_video_test_endpoint():
    """
    Simplified test endpoint for video processing to debug CORS and method issues
    """
    # No need to handle CORS manually - Flask-CORS extension takes care of it
    if request.method == 'OPTIONS':
        return app.make_default_options_response()
    
    # Handle the actual request
    try:
        # Parse JSON data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Extract parameters
        filename = data.get('filename')
        mock_file = data.get('mock_file', False)
        
        if not filename:
            return jsonify({"error": "Filename is required"}), 400
        
        # Just log the request and return success for testing
        print(f"TEST ENDPOINT: Received request to process video: {filename}")
        print(f"TEST ENDPOINT: Mock file requested: {mock_file}")
        
        # Generate a session ID
        session_id = str(uuid.uuid4())[:8]
        
        # Add an entry to sessions.json if mock_file is True
        if mock_file:
            sessions_file = os.path.join(os.path.dirname(__file__), 'sessions.json')
            try:
                if os.path.exists(sessions_file):
                    with open(sessions_file, 'r') as f:
                        sessions = json.load(f)
                else:
                    sessions = {"sessions": []}
                
                # Add new mock session
                sessions["sessions"].append({
                    "id": session_id,
                    "name": f"Test Analysis of {filename}",
                    "created": datetime.now().isoformat(),
                    "is_mock": True
                })
                
                # Save back to the file
                with open(sessions_file, 'w') as f:
                    json.dump(sessions, f, indent=2)
                
                print(f"TEST ENDPOINT: Added mock session with ID: {session_id}")
                
            except Exception as e:
                print(f"TEST ENDPOINT: Error creating mock session: {e}")
        
        # Return success
        return jsonify({
            "success": True,
            "message": "Test endpoint: Processing request received",
            "session_id": session_id
        })
        
    except Exception as e:
        print(f"TEST ENDPOINT: Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

def update_latest_frame(frame, detections):
    """Update the latest frame and detections for HTTP API"""
    global latest_frame, latest_detections
    
    with frame_lock:
        # Store the frame for HTTP API
        if frame is not None:
            try:
                # Create a copy to avoid reference issues
                latest_frame = frame.copy()
            except Exception as e:
                print(f"Error copying frame: {e}")
                # Fallback if copy fails
                latest_frame = frame
        
        # Convert detections to a serializable format
        serializable_detections = []
        for cls_name, conf, (x1, y1, x2, y2) in detections:
            serializable_detections.append({
                "label": cls_name,
                "confidence": float(conf),
                "box": [int(x1), int(y1), int(x2), int(y2)]
            })
        
        latest_detections = serializable_detections

def update_analysis_status(frame_count, unique_frames=None, fps=None):
    """Update the analysis status"""
    global analysis_status
    
    with frame_lock:
        analysis_status["frame_count"] = frame_count
        if unique_frames is not None:
            analysis_status["unique_frames"] = unique_frames
        if fps is not None:
            analysis_status["fps"] = fps

def run_live_recording_thread(camera_id=1, max_api_calls=10, threshold=30):
    """Wrapper function to run live recording in a thread with stop event"""
    global latest_frame, latest_detections
    
    try:
        run_live_recording(camera_id, max_api_calls, threshold)
    except Exception as e:
        print(f"Error in live recording thread: {e}")
    finally:
        # Reset stop event
        stop_live_feed.clear()
        
        # Clean up any remaining resources
        cv2.destroyAllWindows()
        
        # Clean up the frame data
        with frame_lock:
            # Clear the latest frame
            if latest_frame is not None:
                latest_frame = None
            
            # Clear all detections
            latest_detections.clear()
        
        # Ensure analysis status is properly reset
        global analysis_status
        with frame_lock:
            analysis_status["is_running"] = False
            if analysis_status["start_time"]:
                analysis_status["elapsed_time"] = time.time() - analysis_status["start_time"]
            analysis_status["start_time"] = None
            analysis_status["frame_count"] = 0
            analysis_status["unique_frames"] = 0

def run_live_recording(camera_id=1, max_api_calls=10, threshold=30):
    """Run the live camera/screen recording and analysis mode"""
    global analysis_status
    
    print("\n--- Starting Live Recording and Analysis ---")
    
    # Initialize camera
    cap = initialize_camera(camera_id)
    if cap is None:
        return False
    
    # Load YOLO model
    yolo_model = load_yolo_model()
    if yolo_model is None:
        print("Warning: Running without object detection")
        return False
    
    # Set up directories and files for this session
    base_dir = "video_insights"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Create timestamp for this session
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(base_dir, f"live_session_{session_timestamp}")
    os.makedirs(session_dir)
    
    images_dir = os.path.join(session_dir, "images")
    os.makedirs(images_dir)
    
    metadata_file = os.path.join(session_dir, "keyframes.json")
    
    # Update analysis status
    with frame_lock:
        analysis_status["is_running"] = True
        analysis_status["session_dir"] = session_dir
        analysis_status["start_time"] = time.time()
        analysis_status["frame_count"] = 0
        analysis_status["unique_frames"] = 0
        analysis_status["fps"] = 0
    
    # Dictionary to track FPS data
    fps_data = {
        'frame_count': 0,
        'time_elapsed': 0
    }
    
    # Frame counter for controlling YOLO processing frequency
    frame_counter = 0
    
    # Create detection tracker, frame deduplicator and LLM processor - use main4 module
    tracker = main4.DetectionTracker(buffer_size=8, max_age=5)
    deduplicator = main4.FrameDeduplicator(threshold=threshold, 
                                    images_dir=images_dir, 
                                    metadata_file=metadata_file)
    llm_processor = main4.LLMProcessor(max_workers=3, max_api_calls=max_api_calls)
    
    # Record start time for video timestamps
    start_time = time.time()
    
    print(f"\nPress 'q' to stop recording")
    print(f"Session directory: {session_dir}")
    
    # Modified get_frame function to update latest frame for HTTP API
    def get_frame_with_http(cap, fps_data, yolo_model, frame_counter, tracker, deduplicator, llm_processor, start_time):
        # Check if we should stop
        if stop_live_feed.is_set():
            return False
        
        # If analysis is paused, we'll just return True without processing
        if not analysis_status["is_running"]:
            time.sleep(0.1)  # Small sleep to avoid excessive CPU usage
            return True
        
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            return False
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        timestamp = f"{minutes:02d}:{seconds:02d}"
        
        # Process with YOLO every 2 frames for better balance
        process_with_yolo = frame_counter % 2 == 0
        current_detections = []
        
        if process_with_yolo and yolo_model:
            try:
                results = yolo_model(frame)
                
                # Extract detections
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Get class and confidence
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        cls_name = result.names[cls_id]
                        
                        # Only add if confidence is high enough
                        if conf > 0.4:  # Minimum confidence threshold
                            current_detections.append((cls_name, conf, (x1, y1, x2, y2)))
            
            except Exception as e:
                print(f"Error in YOLO processing: {e}")
        
        # Update tracker and get smoothed detections
        smoothed_detections = tracker.update(current_detections)
        
        # Draw smoothed boxes
        display_frame = frame.copy()
        for cls_name, conf, (x1, y1, x2, y2) in smoothed_detections:
            # Draw bounding box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Display class name and confidence
            label = f"{cls_name}: {conf:.2f}"
            cv2.putText(display_frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display timestamp
        cv2.putText(display_frame, timestamp, (10, display_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # If we have detections, create frame data and check for uniqueness
        is_keyframe = False
        if smoothed_detections:
            frame_data = main4.create_frame_data(timestamp, smoothed_detections, elapsed_time)
            is_keyframe, frame_id = deduplicator.add_frame(frame_data, frame)
            
            # Add keyframe indicator if this is a unique frame
            if is_keyframe:
                cv2.putText(display_frame, "KEYFRAME", (display_frame.shape[1] - 150, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                
                # Queue this keyframe for LLM description
                image_path = os.path.join(deduplicator.images_dir, f"{frame_id}.jpg")
                llm_processor.add_frame_for_processing(
                    frame_id,
                    image_path,
                    frame_data["objects"],
                    deduplicator
                )
        
        # Calculate and update FPS
        end_time = time.time()
        fps_data['frame_count'] += 1
        fps_data['time_elapsed'] += (end_time - (start_time + elapsed_time))
        
        # Update FPS calculation every 10 frames
        fps = 0
        if fps_data['frame_count'] % 10 == 0:
            fps = 10 / fps_data['time_elapsed'] if fps_data['time_elapsed'] > 0 else 0
            print(f"FPS: {fps:.2f}")
            fps_data['time_elapsed'] = 0
            
            # Add FPS text to frame
            cv2.putText(display_frame, f"FPS: {fps:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Update the latest frame for HTTP API
        update_latest_frame(display_frame, smoothed_detections)
        
        # Update analysis status
        update_analysis_status(
            frame_counter, 
            len(deduplicator.unique_frames) if is_keyframe else None,
            fps if fps > 0 else None
        )
        
        # Only show OpenCV window when NOT started via API (when started from command line)
        # This prevents the local window from opening when controlled from the web UI
        if stop_live_feed.is_set():  # If this is set, we're being stopped via API
            # Don't show any windows when running via API
            pass
        else:
            # Only show OpenCV window when started from command line (not via API)
            # We can determine this by checking if we're in the main thread (not a daemon thread)
            if not threading.current_thread().daemon:
                cv2.imshow("YOLO Detection", display_frame)
                key = cv2.waitKey(100) 
                if key == ord('q'):
                    return False
        
        # Add a small sleep to control frame rate
        time.sleep(0.01)  # Small sleep to avoid consuming too much CPU
        
        return True
    
    try:
        while True:
            if not get_frame_with_http(cap, fps_data, yolo_model, frame_counter, 
                            tracker, deduplicator, llm_processor, start_time):
                break
            frame_counter += 1
    except KeyboardInterrupt:
        print("\nRecording stopped by user")
    finally:
        # Update analysis status
        with frame_lock:
            analysis_status["is_running"] = False
        
        # Wait for any pending LLM tasks to complete
        print("\nShutting down...")
        llm_processor.shutdown()
        
        # Clean up resources
        cap.release()
        cv2.destroyAllWindows()
        
        # Save final metadata
        deduplicator.save_metadata()
        
        # Print summary
        print(f"\nProcessed {frame_counter} frames")
        print(f"Saved {len(deduplicator.unique_frames)} unique keyframes to {images_dir}")
        print(f"Metadata saved to {metadata_file}")
        print(f"Session directory: {session_dir}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Video Analysis System - Process live camera feeds or video files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Create uploads directory if it doesn't exist
    uploads_dir = os.path.join(os.path.dirname(__file__), 'uploads')
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)
        print(f"Created uploads directory: {uploads_dir}")
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument('--live', action='store_true',
                         help='Record and analyze live camera/screen feed')
    mode_group.add_argument('--video', metavar='VIDEO_PATH',
                         help='Process an existing video file')
    mode_group.add_argument('--server-only', action='store_true',
                         help='Run the Flask server only, without recording or processing')
    
    # Common parameters
    parser.add_argument('-m', '--max-calls', type=int, default=10,
                      help='Maximum number of LLM API calls to make (to control costs)')
    parser.add_argument('-t', '--threshold', type=int, default=30,
                      help='Threshold for detecting significant changes between frames')
    
    # Live mode parameters
    parser.add_argument('-c', '--camera', type=int, default=1,
                      help='Camera ID to use for live recording (default: 1)')
    
    # Video mode parameters
    parser.add_argument('-o', '--output-dir', 
                      help='Directory to save results for video processing')
    parser.add_argument('-i', '--interval', type=int, default=15,
                      help='Process every Nth frame from video file (higher values = faster)')
    
    args = parser.parse_args()
    
    try:
        # Start Flask server in a separate thread
        def run_flask_server():
            app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        
        flask_thread = threading.Thread(target=run_flask_server, daemon=True)
        flask_thread.start()
        
        print(f"HTTP API started at http://localhost:5000")
        print(f"Available endpoints:")
        print(f"  - GET  /api/status               : Current analysis status")
        print(f"  - GET  /api/latest-frame         : Latest frame as JPEG")
        print(f"  - GET  /api/latest-detections    : Latest detections")
        print(f"  - POST /api/control              : Control analysis (pause/resume/start_live/stop_live)")
        print(f"  - GET  /api/sessions             : List all sessions")
        print(f"  - GET  /api/sessions/<id>        : Get session details")
        print(f"  - GET  /api/sessions/<id>/frames : Get frames for a session")
        print(f"  - GET  /api/frames/<id>/image    : Get frame image")
        print(f"  - POST /api/cameras/check         : Check camera availability")
        
        if args.live:
            # Set global variables
            global camera_id, max_api_calls, threshold
            camera_id = args.camera
            max_api_calls = args.max_calls
            threshold = args.threshold
            
            # Run live recording mode directly
            run_live_recording(
                camera_id=camera_id,
                max_api_calls=max_api_calls,
                threshold=threshold
            )
            
        elif args.video:
            # Check if video file exists
            if not os.path.exists(args.video):
                print(f"Error: Video file not found: {args.video}")
                return 1
                
            # Run video processing mode
            process_video_file(
                video_path=args.video,
                output_dir=args.output_dir,
                frame_interval=args.interval,
                max_api_calls=args.max_calls,
                threshold=args.threshold
            )
        
        else:
            # Just run the Flask server
            print("\n--- Running Flask server only mode ---")
            print("Press Ctrl+C to exit")
            print("\nTo start live recording via API, send POST to /api/control with:")
            print('{"command": "start_live", "camera_id": 0}')
            # Keep the main thread alive to allow the Flask thread to run
            while True:
                time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
