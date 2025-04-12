import cv2

def initialize_camera(camera_id=0):
    cap = cv2.VideoCapture(1)

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

cap = initialize_camera()

if cap is not None:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"Camera resolution: {width}x{height}")
    ret, frame = cap.read()

    cv2.imwrite('test.jpg', frame)