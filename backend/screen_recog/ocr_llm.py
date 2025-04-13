import cv2
import pytesseract
import openai
import os
import glob
from dotenv import load_dotenv
import concurrent.futures

# Load environment variables
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------------------------------------
# Function 1: OCR - Extract text from one image
# -----------------------------------------------------------
def extract_text_from_image(image_path):
    frame_id = os.path.splitext(os.path.basename(image_path))[0]
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(thresh).strip()
    return {"frame_id": frame_id, "text": text}

# -----------------------------------------------------------
# Helper: Split a dictionary into chunks
# -----------------------------------------------------------
def chunk_dict(data, chunk_size):
    items = list(data.items())
    for i in range(0, len(items), chunk_size):
        yield dict(items[i:i + chunk_size])

# -----------------------------------------------------------
# Function 2: LLM - Analyze batch of OCR results as forensic JSON
# -----------------------------------------------------------
def describe_batch_keyframes_llm(ocr_results: dict):
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
    - a `"description"` (explain why it’s suspicious in a digital forensics context)
- Finish with a **relevance_to_investigation** statement — description how or why this frame might matter to a forensic analyst.

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
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )
    return response.choices[0].message.content

# -----------------------------------------------------------
# Function 3: Parallel Batching for Faster LLM Calls
# -----------------------------------------------------------
def describe_in_parallel(ocr_results, batch_size=5):
    batches = list(chunk_dict(ocr_results, batch_size))
    results = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(describe_batch_keyframes_llm, batch) for batch in batches]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    return results

# -----------------------------------------------------------
# Function 4: Wrapper - Process keyframes with OCR + LLM
# -----------------------------------------------------------
def process_keyframes_with_llm(image_paths):
    ocr_results = {}
    for path in image_paths:
        result = extract_text_from_image(path)
        ocr_results[result["frame_id"]] = result["text"]

    return describe_in_parallel(ocr_results)

# -----------------------------------------------------------
# Main Execution
# -----------------------------------------------------------
keyframe_paths = sorted(glob.glob("images/*"))
output = process_keyframes_with_llm(keyframe_paths)

for batch_output in output:
    print(batch_output)
