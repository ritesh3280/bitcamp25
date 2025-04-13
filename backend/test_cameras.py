#!/usr/bin/env python3
"""
Camera Test Script

This script helps identify available cameras on your system
"""

import cv2
import argparse
import time

def list_available_cameras(max_id=10):
    """List available camera devices"""
    print("Checking for available cameras...")
    
    available_cameras = []
    
    # Check camera indices from 0 to max_id
    for i in range(max_id):
        try:
            print(f"Testing camera {i}...")
            cap = cv2.VideoCapture(i)
            
            if cap.isOpened():
                # Try to read a frame to confirm it works
                ret, frame = cap.read()
                
                if ret:
                    # Get camera properties
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    camera_info = {
                        "id": i,
                        "name": f"Camera {i}",
                        "width": width,
                        "height": height,
                        "fps": fps
                    }
                    
                    available_cameras.append(camera_info)
                    print(f"  ✅ Found camera {i}: {width}x{height} @ {fps}fps")
                else:
                    print(f"  ❌ Camera {i} opened but couldn't read frame")
                
                cap.release()
            else:
                print(f"  ❌ Camera {i} not available")
                
        except Exception as e:
            print(f"  ❌ Error testing camera {i}: {str(e)}")
    
    return available_cameras

def test_camera(camera_id, display_time=5):
    """Test a specific camera by displaying its feed"""
    print(f"\nTesting camera {camera_id}...")
    
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return False
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Camera {camera_id} opened: {width}x{height} @ {fps}fps")
    print(f"Displaying feed for {display_time} seconds...")
    
    start_time = time.time()
    frame_count = 0
    
    try:
        while time.time() - start_time < display_time:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Display frame
            cv2.imshow(f"Camera {camera_id} Test", frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        
        # Calculate actual FPS
        elapsed = time.time() - start_time
        if elapsed > 0:
            actual_fps = frame_count / elapsed
            print(f"Captured {frame_count} frames in {elapsed:.2f} seconds")
            print(f"Actual FPS: {actual_fps:.2f}")
        
        return True

def main():
    parser = argparse.ArgumentParser(description='Test camera devices')
    parser.add_argument('--list', action='store_true', help='List available cameras')
    parser.add_argument('--test', type=int, help='Test a specific camera by ID')
    parser.add_argument('--time', type=int, default=5, help='Display time in seconds for testing')
    parser.add_argument('--max', type=int, default=10, help='Maximum camera ID to check')
    
    args = parser.parse_args()
    
    if args.list:
        cameras = list_available_cameras(args.max)
        
        if cameras:
            print(f"\nFound {len(cameras)} camera(s):")
            for cam in cameras:
                print(f"  Camera {cam['id']}: {cam['width']}x{cam['height']} @ {cam['fps']}fps")
        else:
            print("\nNo cameras found.")
            print("Possible reasons:")
            print("  - No cameras connected")
            print("  - Camera is already in use by another application")
            print("  - Need system permissions to access camera")
    
    if args.test is not None:
        test_camera(args.test, args.time)
    
    if not args.list and args.test is None:
        # If no options specified, list and then test first available
        cameras = list_available_cameras(args.max)
        
        if cameras:
            print(f"\nTesting first available camera (ID {cameras[0]['id']})...")
            test_camera(cameras[0]['id'], args.time)
        else:
            print("\nNo cameras available to test.")

if __name__ == '__main__':
    main() 