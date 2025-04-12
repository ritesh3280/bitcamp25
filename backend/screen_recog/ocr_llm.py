import cv2
import pytesseract
import openai
import os
import glob
from dotenv import load_dotenv

# Load environment variables (e.g., your OpenAI API key)
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------------------------------------
# Function 1: OCR - Extract text from one image
# -----------------------------------------------------------
def extract_text_from_image(image_path):
    """
    Runs OCR on a given image and returns a dict with frame ID and extracted text.
    """
    frame_id = os.path.splitext(os.path.basename(image_path))[0]

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(thresh).strip()

    return {
        "frame_id": frame_id,
        "text": text
    }

# -----------------------------------------------------------
# Function 2: LLM - Analyze batch of OCR results as forensic JSON
# -----------------------------------------------------------
def describe_batch_keyframes_llm(ocr_results: dict):
    """
    Given a dict of {frame_id: text}, sends to LLM for JSON-based screen analysis.
    Returns JSON string with one object per frame.
    """
    prompt_entries = "\n".join(
        f"Frame ID: {frame_id}\nText:\n\"\"\"{text}\"\"\"\n"
        for frame_id, text in ocr_results.items()
    )

    prompt = f"""
You are a forensic AI assistant helping cybercrime investigators analyze screenshots taken from a suspect's computer screen.

Each frame has been processed using OCR and contains text extracted from the screen. You are provided with multiple frame IDs and their respective texts.

For each frame:
- Provide a **brief screen_summary** of what the screen appears to show.
- List upto **5 unique potential_indicators** (e.g., commands, tools, IP addresses, usernames, logs, software names, file paths, system info).
- Identify up to **3 unique red_flags**, if any. These are suspicious or anomalous elements that might suggest malicious activity.
  - If **no red flags** are found, leave the list empty (`"red_flags": []`)
  - If present, each red flag must include:
    - a `"flag"` (short title)
    - a `"description"` (why it’s suspicious in a digital forensics context)
- Finish with a **relevance_to_investigation** statement — explain how or why this frame might matter to a forensic analyst.

Respond ONLY with valid JSON in the following format:

{{
  "frames": [
    {{
      "frame_id": "frame_001",
      "screen_summary": "Brief description of what the screen shows.",
      "potential_indicators": ["List of indicators"],
      "red_flags": [
        {{
          "flag": "Red flag title",
          "description": "Why it's suspicious"
        }}
      ],
      "relevance_to_investigation": "Explanation of investigative relevance"
    }}
  ]
}}

If no red flags are found, set `"red_flags": []` for that frame.

Here are the screen texts:
{prompt_entries}

Respond ONLY with valid JSON.
"""


    response = client.chat.completions.create(
        model="gpt-4o",  # Change to "gpt-4" if needed
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )

    return response.choices[0].message.content

# -----------------------------------------------------------
# Function 3: Wrapper - Process keyframes with OCR + LLM
# -----------------------------------------------------------
def process_keyframes_with_llm(image_paths):
    """
    Given a list of image paths (keyframes), this:
    1. Extracts OCR text for each
    2. Sends the batch to an LLM for forensic screen analysis
    3. Returns JSON response from the LLM
    """
    ocr_results = {}
    for path in image_paths:
        result = extract_text_from_image(path)
        ocr_results[result["frame_id"]] = result["text"]

    return describe_batch_keyframes_llm(ocr_results)

keyframe_paths = sorted(glob.glob("images/*"))

output = process_keyframes_with_llm(keyframe_paths)
print(output)


