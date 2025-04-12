from openai import OpenAI
from PIL import Image
import base64
import io
import os
import json
from datetime import datetime
from dotenv import load_dotenv
import concurrent.futures

# Load environment variables
load_dotenv()
client = OpenAI()  # Uses OPENAI_API_KEY from .env

# -----------------------------------
# Encode a single image to base64
# -----------------------------------
def encode_image_to_base64(image_path):
    try:
        with Image.open(image_path) as img:
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

# -----------------------------------
# Describe one image (same as before)
# -----------------------------------
def describe_frame_with_llm(image_path, object_identified=""):
    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        return {"frame": os.path.basename(image_path), "error": "Encoding failed"}

    prompt = (
        f"This frame contains a {object_identified}. "
        f"Provide a brief forensic summary focusing on this object and its immediate context. "
        f"Limit to 2-3 sentences." if object_identified else
        "Provide a brief forensic summary of this frame. Limit to 2-3 sentences and focus only on the most relevant details."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a forensic video analyst. Provide short, focused summaries of what is seen in the frame. Prioritize key investigative details. Be concise (2-3 sentences max)."
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
            "frame": os.path.basename(image_path),
            "full_path": image_path,
            "object_detected": object_identified,
            "description": response.choices[0].message.content,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        return {
            "frame": os.path.basename(image_path),
            "full_path": image_path,
            "error": f"OpenAI API error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

# -----------------------------------
# Parallel Processing of Many Frames
# -----------------------------------
def process_frames_in_parallel(image_paths, object_mapping=None, max_workers=5):
    """
    image_paths: list of image paths
    object_mapping: optional dict {image_path: "object detected"}
    """
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for path in image_paths:
            obj = object_mapping.get(path, "") if object_mapping else ""
            futures.append(executor.submit(describe_frame_with_llm, path, obj))

        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    return results

# -----------------------------------
# Example Use Case
# -----------------------------------
if __name__ == "__main__":
    import json
    import glob
    import os

    # Load the JSON object mapping
    with open("object_mapping.json", "r") as f:
        raw_object_mapping = json.load(f)

    # Normalize keys from JSON
    object_info = {
        os.path.normpath(k): ", ".join(v) if isinstance(v, list) else str(v)
        for k, v in raw_object_mapping.items()
    }

    # Collect all JPEG images and normalize paths
    image_files = sorted(glob.glob("images/*.jpeg"))
    image_files = [os.path.normpath(f) for f in image_files]

    # Debug: print file matching info
    print(f"Found {len(image_files)} image(s) in 'images/' folder.")
    unmatched_files = [f for f in image_files if f not in object_info]
    if unmatched_files:
        print("\n⚠️ Warning: Some image files have no object mapping:")
        for f in unmatched_files:
            print(" -", f)

    # Filter only those with metadata
    image_files = [f for f in image_files if f in object_info]

    print(f"\n✅ Ready to process {len(image_files)} image(s)...")

    # Run analysis
    results = process_frames_in_parallel(image_files, object_info)

    # Print outputs
    for frame_result in results:
        print("\n--- Frame ---")
        print(f"Image: {frame_result.get('frame')}")
        print(f"Object(s): {frame_result.get('object_detected')}")
        print(f"Description: {frame_result.get('description') or frame_result.get('error')}")
