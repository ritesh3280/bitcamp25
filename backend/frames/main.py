import cv2

def initialize_camera(camera_id=0):
    cap = cv2.VideoCapture(camera_id)

    if cap.isOpened():

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

x = 0;

def getFrame(cap):
    global x;
    ret,frame = cap.read();
    cv2.imwrite(f"{x}.jpg",frame);
    x+=1;

