import cv2
import os
import uuid

# Create directories if not exist
POS_PATH = os.path.join("data", "positive")
ANC_PATH = os.path.join("data", "anchor")
os.makedirs(POS_PATH, exist_ok=True)
os.makedirs(ANC_PATH, exist_ok=True)

# Open camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Crop the region of interest (you can adjust as needed)
    cropped_frame = frame[120:120+250, 200:200+250]

    # Show on screen
    cv2.imshow("Web Cam", cropped_frame)

    # Key press logic
    key = cv2.waitKey(1) & 0xFF

    if key == ord('a'):
        imgname = os.path.join(ANC_PATH, f"{uuid.uuid1()}.jpg")
        cv2.imwrite(imgname, cropped_frame)
        print(f"Saved anchor image: {imgname}")

    elif key == ord('p'):
        imgname = os.path.join(POS_PATH, f"{uuid.uuid1()}.jpg")
        cv2.imwrite(imgname, cropped_frame)
        print(f"Saved positive image: {imgname}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
