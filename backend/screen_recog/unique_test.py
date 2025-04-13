import json
import math

# === Sample JSON ===
sample_json = [
    {
        "timestamp": "00:01",
        "objects": ["usb", "screen"],
        "boxes": {
            "usb": [100, 150, 40, 40],
            "screen": [300, 200, 200, 150]
        }
    },
    {
        "timestamp": "00:02",
        "objects": ["usb", "screen"],
        "boxes": {
            "usb": [102, 152, 41, 41],
            "screen": [300, 200, 200, 150]
        }
    },
    {
        "timestamp": "00:05",
        "objects": ["usb", "screen", "hard drive"],
        "boxes": {
            "usb": [105, 160, 42, 42],
            "screen": [300, 200, 200, 150],
            "hard drive": [450, 300, 100, 100]
        }
    }
]

def box_distance(b1, b2):
    # Euclidean distance of top-left corner + size difference
    return math.sqrt((b1[0]-b2[0])**2 + (b1[1]-b2[1])**2) + abs(b1[2]-b2[2]) + abs(b1[3]-b2[3])

def has_significant_change(curr, prev, threshold=30):
    # Check object set change
    if set(curr['objects']) != set(prev['objects']):
        return True

    # Check position/size of shared objects
    for obj in curr['boxes']:
        if obj not in prev['boxes']:
            return True
        dist = box_distance(curr['boxes'][obj], prev['boxes'][obj])
        if dist > threshold:
            return True

    return False

def deduplicate_frames(frames):
    if not frames:
        return []

    deduped = [frames[0]]
    for i in range(1, len(frames)):
        if has_significant_change(frames[i], deduped[-1]):
            deduped.append(frames[i])
    return deduped

# === Run Deduplication ===
deduped_frames = deduplicate_frames(sample_json)

# Print results
print("Unique frames (based on object/content change):\n")
for frame in deduped_frames:
    print(f"⏱ {frame['timestamp']} — Objects: {frame['objects']}")
