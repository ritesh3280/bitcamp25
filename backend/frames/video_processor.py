#!/usr/bin/env python3
"""
Video Processor - Extract key frames from videos with object detection and LLM descriptions

This script processes video files, identifying unique frames based on detected objects
and generating AI descriptions for each significant frame.
"""

import os
import sys
import argparse
from main4 import process_video_file

def main():
    parser = argparse.ArgumentParser(
        description='Process videos with object detection and LLM descriptions',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('video_path', 
                        help='Path to the video file to process')
    
    parser.add_argument('-o', '--output-dir', 
                        help='Directory to save results (default: creates directory based on video name)')
    
    parser.add_argument('-i', '--interval', type=int, default=15,
                        help='Process every Nth frame (higher values = faster processing)')
    
    parser.add_argument('-m', '--max-calls', type=int, default=10,
                        help='Maximum number of LLM API calls to make (to control costs)')
    
    parser.add_argument('-t', '--threshold', type=int, default=30,
                        help='Threshold for detecting significant changes between frames')
    
    args = parser.parse_args()
    
    # Validate video file exists
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return 1
    
    # Process video
    try:
        output_dir = process_video_file(
            video_path=args.video_path,
            output_dir=args.output_dir,
            frame_interval=args.interval,
            max_api_calls=args.max_calls,
            threshold=args.threshold
        )
        
        if output_dir:
            print(f"\nSuccess! Video analysis complete.")
            print(f"Open {output_dir}/keyframes.json to view metadata and descriptions")
        else:
            print("Processing failed. Check error messages above.")
            return 1
            
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        return 1
    except Exception as e:
        print(f"Error processing video: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())