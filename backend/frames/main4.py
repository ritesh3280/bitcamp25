import cv2
import time
import numpy as np
import math
import json
import os
import uuid
from ultralytics import YOLO
from collections import deque
from datetime import datetime
import base64
import io
from PIL import Image
import concurrent.futures
from dotenv import load_dotenv
from openai import OpenAI


# Load environment variables for OpenAI
load_dotenv()
client = OpenAI()  # Uses OPENAI_API_KEY from .env

class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that can handle NumPy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyJSONEncoder, self).default(obj)


def encode_image_to_base64(image_path):
    """Convert an image file to base64 encoding"""
    try:
        with Image.open(image_path) as img:
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None


def describe_frame_with_llm(image_path, objects_identified=None):
    """Get LLM description of an image with detected objects"""
    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        return {"error": "Encoding failed"}
    
    # Create enhanced prompt that focuses on electronic device connections
    electronic_devices = [
        "keyboard", "mouse", "laptop", "desktop", "computer", "monitor", 
        "tv", "tvmonitor", "display", "cell phone", "tablet", "usb", 
        "hard drive", "hdmi", "cable", "router", "modem", "server", "charger"
    ]
    
    # Check if we have electronic devices in the objects
    has_electronics = False
    if objects_identified:
        has_electronics = any(obj in electronic_devices for obj in objects_identified)
    
    # Create a focused prompt based on detected objects    
    if objects_identified:
        object_text = ", ".join(objects_identified)
        if has_electronics:
            # Enhanced electronics-focused prompt
            prompt = (
                f"This frame contains the following electronic devices/objects: {object_text}. "
                f"Provide a detailed forensic analysis focusing on: "
                f"1) What electronic devices are visible "
                f"2) Whether devices are powered on/off "
                f"3) What connections exist between devices (cables, wireless) "
                f"4) Any peripherals connected to main devices "
                f"5) Status of device screens (what's displayed) if visible "
                f"Be specific and detailed but limit to 3-4 sentences maximum."
            )
        else:
            # Standard prompt for non-electronic objects
            prompt = (
                f"This frame contains: {object_text}. "
                f"Provide a brief forensic summary focusing on these objects and their immediate context. "
                f"Limit to 2-3 sentences."
            )
    else:
        # Generic prompt when no objects are specified
        prompt = (
            "Provide a brief forensic summary of this frame. "
            "If electronic devices are visible, describe their state, connections, "
            "and what's displayed on screens. Limit to 2-3 sentences."
        )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a forensic digital evidence analyst specializing in electronic devices. "
                               "Provide detailed analysis of devices, their connections, power state, and screen content. "
                               "Focus on technical details that could be relevant to digital forensics. Be concise but thorough."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]
                }
            ],
        )
        return {
            "description": response.choices[0].message.content,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        print(f"LLM API error: {str(e)}")
        return {
            "error": f"LLM API error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }


def process_video_file(video_path, output_dir=None, frame_interval=15, max_api_calls=10, threshold=30):
    """
    Process a video file through the object detection and LLM description pipeline
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save results (default: creates directory based on video filename)
        frame_interval: Only process every Nth frame (default: 15, which means 2 frames/sec for 30fps video)
        max_api_calls: Maximum number of LLM API calls to make (default: 10)
        threshold: Threshold for detecting significant changes between frames (default: 30)
        
    Returns:
        Path to the session directory containing results
    """
    # Create session directory
    if output_dir is None:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        base_dir = "video_insights"
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        
        # Create timestamp for this session
        session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(base_dir, f"{video_name}_{session_timestamp}")
        os.makedirs(session_dir)
    else:
        session_dir = output_dir
        if not os.path.exists(session_dir):
            os.makedirs(session_dir)
    
    # Create images directory
    images_dir = os.path.join(session_dir, "images")
    os.makedirs(images_dir)
    
    # Setup metadata file
    metadata_file = os.path.join(session_dir, "keyframes.json")
    
    # Load YOLO model
    yolo_model = load_yolo_model()
    if yolo_model is None:
        print("Error: Failed to load YOLO model")
        return None
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video: {video_path}")
    print(f"FPS: {fps:.2f}, Total frames: {total_frames}, Duration: {duration:.2f}s")
    print(f"Processing every {frame_interval} frames ({fps/frame_interval:.2f} frames/sec)")
    
    # Create tracker and deduplicator
    tracker = DetectionTracker(buffer_size=8, max_age=5)
    deduplicator = FrameDeduplicator(threshold=threshold, images_dir=images_dir, metadata_file=metadata_file)
    llm_processor = LLMProcessor(max_workers=3, max_api_calls=max_api_calls)
    
    # Process video
    frame_count = 0
    processed_count = 0
    keyframe_count = 0
    
    print("\nProcessing video...")
    
    # Add video metadata
    deduplicator.all_keyframes_data["metadata"].update({
        "video_source": video_path,
        "fps": float(fps),
        "total_frames": total_frames,
        "duration": float(duration),
        "frame_interval": frame_interval
    })
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only process every Nth frame to improve performance
            if frame_count % frame_interval == 0:
                # Calculate timestamp from frame position
                elapsed_time = frame_count / fps if fps > 0 else 0
                minutes = int(elapsed_time // 60)
                seconds = int(elapsed_time % 60)
                timestamp = f"{minutes:02d}:{seconds:02d}"
                
                # Process frame with YOLO
                try:
                    results = yolo_model(frame)
                    current_detections = []
                    
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
                    print(f"Error in YOLO processing frame {frame_count}: {e}")
                    current_detections = []
                
                # Update tracker and get smoothed detections
                smoothed_detections = tracker.update(current_detections)
                
                # If we have detections, create frame data
                if smoothed_detections:
                    frame_data = create_frame_data(timestamp, smoothed_detections, elapsed_time)
                    is_keyframe, frame_id = deduplicator.add_frame(frame_data, frame)
                    
                    if is_keyframe:
                        keyframe_count += 1
                        progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                        print(f"Keyframe {keyframe_count} detected at {timestamp} ({progress:.1f}% through video)")
                        
                        # Queue for LLM processing
                        image_path = os.path.join(images_dir, f"{frame_id}.jpg")
                        llm_processor.add_frame_for_processing(
                            frame_id,
                            image_path,
                            frame_data["objects"],
                            deduplicator
                        )
                
                processed_count += 1
                
                # Print progress every 100 processed frames
                if processed_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    print(f"Processed {processed_count} frames ({progress:.1f}% of video)")
                
            frame_count += 1
            
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    finally:
        # Wait for any pending LLM tasks
        print("\nFinishing up LLM processing...")
        llm_processor.shutdown()
        
        # Release resources
        cap.release()
        
        # Save final metadata
        deduplicator.save_metadata()
        
        # Print summary
        print(f"\nProcessing complete!")
        print(f"Processed {processed_count} frames from {total_frames} total frames ({frame_count/frame_interval:.1f}%)")
        print(f"Identified {keyframe_count} unique keyframes")
        print(f"Results saved to {session_dir}")
        
        return session_dir


def initialize_camera(camera_id=1):
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        print("Available cameras:")
        # List available cameras (may vary by system)
        for i in range(8):
            temp_cap = cv2.VideoCapture(i)
            if temp_cap.isOpened():
                print(f"  Camera ID {i} is available")
                temp_cap.release()
        return None
    return cap


def load_yolo_model():
    # Load YOLOv11n model
    try:
        model = YOLO('yolov11n.pt')  # Load the model
        print("YOLOv11n model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None


class LLMProcessor:
    def __init__(self, max_workers=3, max_api_calls=10, importance_threshold=0.6):
        self.max_workers = max_workers
        self.processing_queue = []
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.api_call_count = 0  # Track number of API calls made
        self.max_api_calls = max_api_calls  # Maximum number of API calls allowed
        self.importance_threshold = importance_threshold  # Minimum score to process frame
        self.high_value_devices = [
            "keyboard", "mouse", "laptop", "desktop computer", "computer", 
            "monitor", "tv", "tvmonitor", "display", "cell phone", "tablet", 
            "usb", "hard drive", "hdmi", "cable", "router", "modem", "server"
        ]
        self.processed_frames = set()  # Track which frames we've already processed
        
    def calculate_frame_importance(self, frame_id, objects, boxes):
        """Calculate an importance score for a frame based on content"""
        # Base score starts at 0.3
        score = 0.3
        
        # More objects means more information to analyze
        object_count_factor = min(len(objects) / 5, 0.3)  # Up to 0.3 for 5+ objects
        score += object_count_factor
        
        # Presence of high-value devices increases importance
        high_value_count = sum(1 for obj in objects if obj in self.high_value_devices)
        high_value_factor = min(high_value_count / 3, 0.3)  # Up to 0.3 for 3+ high-value devices
        score += high_value_factor
        
        # Multiple electronics in frame suggests possible connections
        if high_value_count >= 2:
            score += 0.2  # Bonus for potential connections between devices
            
        # Large objects (taking up significant screen space) are more important
        large_objects = 0
        for obj in boxes.values():
            # Area calculation (width * height)
            area = obj[2] * obj[3]  
            if area > 10000:  # Arbitrary threshold for "large" object
                large_objects += 1
        
        large_object_factor = min(large_objects / 2, 0.2)  # Up to 0.2 for 2+ large objects
        score += large_object_factor
        
        # Cap the score at 1.0
        score = min(score, 1.0)
        
        return score
    
    def add_frame_for_processing(self, frame_id, image_path, objects, deduplicator):
        """Queue a frame for LLM description if it meets importance criteria"""
        # Check if we've reached the maximum number of allowed API calls
        if self.api_call_count >= self.max_api_calls:
            print(f"⚠️ Maximum API call limit reached ({self.max_api_calls}). Skipping LLM processing for frame {frame_id}.")
            return False
            
        # Check if this frame ID has already been processed
        if frame_id in self.processed_frames:
            return False
        
        # Get the frame data from the deduplicator to assess importance
        frame_data = None
        for frame in deduplicator.all_keyframes_data["frames"]:
            if frame.get("frame_id") == frame_id:
                frame_data = frame
                break
                
        if not frame_data:
            print(f"⚠️ Could not find data for frame {frame_id}")
            return False
            
        # Calculate importance score
        importance = self.calculate_frame_importance(
            frame_id, 
            frame_data.get("objects", []), 
            frame_data.get("boxes", {})
        )
        
        # Skip frames below the importance threshold
        if importance < self.importance_threshold:
            print(f"ℹ️ Frame {frame_id} scored {importance:.2f}, below threshold {self.importance_threshold:.2f}, skipping LLM")
            return False
            
        # Add the frame to processed set
        self.processed_frames.add(frame_id)
            
        # Increment the API call counter
        self.api_call_count += 1
        
        # Submit the task to the executor
        future = self.executor.submit(
            self._process_frame, 
            frame_id, 
            image_path, 
            objects, 
            deduplicator
        )
        self.processing_queue.append(future)
        
        # Print importance
        print(f"🔍 Frame {frame_id} scored {importance:.2f}, sending to LLM ({self.api_call_count}/{self.max_api_calls})")
        
        # Clean up completed tasks
        self._cleanup_completed_tasks()
        
        # If we've reached the limit, print a notification
        if self.api_call_count >= self.max_api_calls:
            print(f"\n🛑 API call limit of {self.max_api_calls} reached. No more frames will be processed.")
            print(f"This is a cost-saving measure for testing purposes.")
            
        return True
    
    def _process_frame(self, frame_id, image_path, objects, deduplicator):
        """Process a single frame with LLM"""
        print(f"Getting LLM description for frame {frame_id}...")
        description = describe_frame_with_llm(image_path, objects)
        
        # Update metadata with description
        deduplicator.add_llm_description(frame_id, description)
        
        print(f"✅ Frame {frame_id} LLM description complete")
        return frame_id, description
    
    def _cleanup_completed_tasks(self):
        """Clean up completed future tasks"""
        self.processing_queue = [f for f in self.processing_queue if not f.done()]
    
    def shutdown(self):
        """Wait for all pending tasks and shut down the executor"""
        print(f"Waiting for {len(self.processing_queue)} pending LLM tasks...")
        concurrent.futures.wait(self.processing_queue)
        self.executor.shutdown()


class DetectionTracker:
    def __init__(self, buffer_size=5, max_age=5):
        self.detections = {}  # {object_id: deque of detections}
        self.buffer_size = buffer_size
        self.next_id = 0
        self.allowed_classes = ["flag", "fan", "keyboard", "mouse", "laptop", "tvmonitor", "display", "monitor", "cell phone", "tablet", "tv", "desktop computer", "computer", "notebook", "printer", "camera", "screen"]
        self.last_seen = {}  # {object_id: frames_since_last_seen}
        self.max_age = max_age  # Maximum frames to keep an object without detection
        
    def update(self, new_detections):
        # new_detections is a list of (cls_name, conf, (x1, y1, x2, y2))
        
        # Filter only allowed classes
        new_detections = [d for d in new_detections if d[0] in self.allowed_classes]
        
        # Increase age counter for all existing detections
        for obj_id in self.detections:
            if obj_id in self.last_seen:
                self.last_seen[obj_id] += 1
            else:
                self.last_seen[obj_id] = 0
        
        # If this is the first frame with detections
        if not self.detections and new_detections:
            for det in new_detections:
                self.detections[self.next_id] = deque([det], maxlen=self.buffer_size)
                self.last_seen[self.next_id] = 0
                self.next_id += 1
            return self._get_smoothed_detections()
        
        # Match new detections with existing ones based on IoU
        matched_ids = set()
        unmatched_detections = []
        
        for det in new_detections:
            cls_name, conf, (x1, y1, x2, y2) = det
            best_iou = 0.3  # Minimum IoU threshold
            best_id = None
            
            for obj_id, history in self.detections.items():
                if obj_id in matched_ids:
                    continue
                    
                last_det = history[-1]
                last_cls, last_conf, (last_x1, last_y1, last_x2, last_y2) = last_det
                
                # Only match with same class
                if cls_name != last_cls:
                    continue
                
                # Calculate IoU
                x_left = max(x1, last_x1)
                y_top = max(y1, last_y1)
                x_right = min(x2, last_x2)
                y_bottom = min(y2, last_y2)
                
                if x_right < x_left or y_bottom < y_top:
                    iou = 0.0
                else:
                    intersection = (x_right - x_left) * (y_bottom - y_top)
                    area1 = (x2 - x1) * (y2 - y1)
                    area2 = (last_x2 - last_x1) * (last_y2 - last_y1)
                    iou = intersection / float(area1 + area2 - intersection)
                
                if iou > best_iou:
                    best_iou = iou
                    best_id = obj_id
            
            if best_id is not None:
                self.detections[best_id].append(det)
                self.last_seen[best_id] = 0  # Reset age counter for matched objects
                matched_ids.add(best_id)
            else:
                unmatched_detections.append(det)
        
        # Add new detections that weren't matched
        for det in unmatched_detections:
            self.detections[self.next_id] = deque([det], maxlen=self.buffer_size)
            self.last_seen[self.next_id] = 0
            self.next_id += 1
        
        # Remove stale tracks (those not seen for max_age frames)
        stale_ids = [obj_id for obj_id in self.detections 
                    if obj_id not in matched_ids and self.last_seen[obj_id] >= self.max_age]
        
        for obj_id in stale_ids:
            del self.detections[obj_id]
            del self.last_seen[obj_id]
        
        return self._get_smoothed_detections()
    
    def _get_smoothed_detections(self):
        result = []
        for obj_id, history in self.detections.items():
            if not history:
                continue
                
            # Get class and confidence from latest detection
            cls_name, conf, _ = history[-1]
            
            # Average the box coordinates
            coords = np.array([box for _, _, box in history])
            avg_box = tuple(np.mean(coords, axis=0).astype(int))
            
            result.append((cls_name, conf, avg_box))
        
        return result


class FrameDeduplicator:
    def __init__(self, threshold=30, images_dir="keyframe_images", metadata_file="keyframes.json"):
        self.frame_data_buffer = []
        self.unique_frames = []
        self.threshold = threshold
        self.images_dir = images_dir
        self.metadata_file = metadata_file
        self.all_keyframes_data = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "version": "1.0"
            },
            "frames": []
        }
        
        # Create save directory if it doesn't exist
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
            print(f"Created directory for keyframe images: {images_dir}")
    
    def box_distance(self, b1, b2):
        # Euclidean distance of top-left corner + size difference
        return math.sqrt((b1[0]-b2[0])**2 + (b1[1]-b2[1])**2) + abs(b1[2]-b2[2]) + abs(b1[3]-b2[3])

    def has_significant_change(self, curr, prev):
        # Check object set change
        if set(curr['objects']) != set(prev['objects']):
            return True

        # Check position/size of shared objects
        for obj in curr['boxes']:
            if obj not in prev['boxes']:
                return True
            dist = self.box_distance(curr['boxes'][obj], prev['boxes'][obj])
            if dist > self.threshold:
                return True

        return False
    
    def add_frame(self, frame_data, frame_image):
        """Add a frame to the buffer and check if it's unique"""
        self.frame_data_buffer.append(frame_data)
        
        # If this is the first frame or it represents a significant change
        if len(self.unique_frames) == 0 or self.has_significant_change(frame_data, self.unique_frames[-1]):
            # Generate a unique ID for this keyframe
            frame_id = str(uuid.uuid4())[:8]
            
            # Add ID to frame data
            frame_data["frame_id"] = frame_id
            frame_data["image_path"] = os.path.join(self.images_dir, f"{frame_id}.jpg")
            
            # Save frame data and image
            self.unique_frames.append(frame_data)
            self.save_keyframe(frame_data, frame_image, frame_id)
            self.all_keyframes_data["frames"].append(frame_data)
            
            # Save the master metadata file after each keyframe
            self.save_metadata()
            
            print(f"Keyframe detected! ID: {frame_id}, Time: {frame_data['timestamp']}, Objects: {frame_data['objects']}")
            return True, frame_id
        
        return False, None
    
    def save_keyframe(self, frame_data, frame_image, frame_id):
        """Save the keyframe image"""
        # Save image
        image_path = os.path.join(self.images_dir, f"{frame_id}.jpg")
        cv2.imwrite(image_path, frame_image)
    
    def save_metadata(self):
        """Save all keyframe metadata to a single JSON file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.all_keyframes_data, f, indent=2, cls=NumpyJSONEncoder)
    
    def add_llm_description(self, frame_id, description_data):
        """Add LLM description to a keyframe's metadata"""
        for frame in self.all_keyframes_data["frames"]:
            if frame.get("frame_id") == frame_id:
                frame["llm_description"] = description_data
                self.save_metadata()
                break


def create_frame_data(timestamp, smoothed_detections, elapsed_time):
    """Convert detection results to the required JSON format"""
    frame_data = {
        "timestamp": timestamp,
        "elapsed_time": elapsed_time,
        "objects": [],
        "boxes": {}
    }
    
    # Add detections to frame data
    for cls_name, conf, (x1, y1, x2, y2) in smoothed_detections:
        if cls_name not in frame_data["objects"]:
            frame_data["objects"].append(cls_name)
        
        # Store as [x, y, width, height] format
        width = x2 - x1
        height = y2 - y1
        frame_data["boxes"][cls_name] = [x1, y1, width, height]
        
        # Also store confidence (not in original format but useful)
        if "confidence" not in frame_data:
            frame_data["confidence"] = {}
        frame_data["confidence"][cls_name] = float(conf)
    
    return frame_data


def get_frame(cap, fps_data, yolo_model, frame_counter, tracker, deduplicator, llm_processor, start_time):
    """Process a frame with object detection and deduplication"""
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
    if smoothed_detections:
        frame_data = create_frame_data(timestamp, smoothed_detections, elapsed_time)
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
    if fps_data['frame_count'] % 10 == 0:
        fps = 10 / fps_data['time_elapsed'] if fps_data['time_elapsed'] > 0 else 0
        print(f"FPS: {fps:.2f}")
        fps_data['time_elapsed'] = 0
        
        # Add FPS text to frame
        cv2.putText(display_frame, f"FPS: {fps:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("YOLO Detection", display_frame)
    
    # Wait for 100ms and check if 'q' key was pressed
    key = cv2.waitKey(100) 
    if key == ord('q'):
        return False
    return True