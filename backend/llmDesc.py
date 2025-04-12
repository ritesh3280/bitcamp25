from openai import OpenAI
from PIL import Image
import base64
import io
import json
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
client = OpenAI()  # Uses the OPENAI_API_KEY from your .env file

def encode_image_to_base64(image_path):
    """Encodes an image to base64 for GPT-4 Vision input."""
    try:
        with Image.open(image_path) as img:
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

def generate_frame_description(frame_data):
    """Calls GPT-4 Turbo Vision to generate a concise description of a frame."""
    image_path = frame_data["frame_id"]
    object_identified = frame_data.get("objects_identified", "")

    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        return "Error: Could not process image"

    # Prompt for short summary
    if object_identified:
        prompt = f"This frame contains a {object_identified}. Provide a brief forensic summary focusing on this object and its immediate context. Limit to 2-3 sentences."
    else:
        prompt = "Provide a brief forensic summary of this frame. Limit to 2-3 sentences and focus only on the most relevant details."

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
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        }
                    ],
                }
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return f"Error generating description: {str(e)}"

def process_frame_direct(image_path, object_identified=None):
    """Processes a single frame directly without using JSON files."""
    frame_data = {
        "frame_id": image_path,
        "objects_identified": object_identified or "",
        "timestamp": datetime.now().isoformat()
    }

    print(f"Processing frame: {image_path}")

    description = generate_frame_description(frame_data)

    frame_output = {
        "frame": os.path.basename(image_path),
        "full_path": image_path,
        "object_detected": object_identified or "",
        "description": description,
        "timestamp": frame_data["timestamp"],
        "processed_at": datetime.now().isoformat()
    }

    print("\n--- Frame Analysis Output ---")
    print(json.dumps(frame_output, indent=2))
    return frame_output

# Example usage
if __name__ == "__main__":
    print("LLM Frame Description module (no file I/O) is ready.")
    print("Use process_frame_direct('image.jpg', 'object name') to run a quick analysis.\n")

    # Example run
    process_frame_direct("llmDescPic.jpeg", "phone and credit card")
