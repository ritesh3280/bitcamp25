"""
Example of using the video processing function

This script shows how to process a video file programmatically
using the new process_video_file function.
"""

import os
from main4 import process_video_file

def main():
    # Path to your video file
    video_file = "path/to/your/video.mp4"  # Update this with your actual video file
    
    # Check if the file exists
    if not os.path.exists(video_file):
        print(f"Error: Video file not found: {video_file}")
        return
    
    print("Starting video analysis...")
    
    # Process the video with custom settings
    output_dir = process_video_file(
        video_path=video_file,
        output_dir=None,  # Auto-create directory based on video filename
        frame_interval=15,  # Process every 15th frame (adjust based on video length/fps)
        max_api_calls=10,  # Limit LLM calls to 10 for testing
        threshold=30  # Sensitivity for detecting unique frames
    )
    
    if output_dir:
        print(f"\nAnalysis complete! Results saved to: {output_dir}")
        
        # Now you can use the keyframes.json metadata for your own purposes
        metadata_file = os.path.join(output_dir, "keyframes.json")
        print(f"You can load the keyframes metadata from: {metadata_file}")
        
        # Example of how you might use the results
        print("\nNext steps:")
        print("1. Load the keyframes.json file")
        print("2. Use the LLM descriptions for search functionality")
        print("3. Display the keyframes and metadata in your application")

if __name__ == "__main__":
    main()