# June 15th, 2024

import os

dirname = os.path.dirname(__file__)
os.chdir(dirname)
model=os.path.join(dirname, "runs", "detect", "train7", "weights", "best.pt")
test=os.path.join(dirname, "realtest")

if not os.path.isfile(model):
    raise FileNotFoundError(model)

from ultralytics import YOLO
import cv2

# Load the YOLOv8 model
model = YOLO(model)

# Open the cam
cap = cv2.VideoCapture(0)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        wkey = cv2.waitKey(1) & 0xFF

        # Break the loop if 'q' is pressed
        if wkey == ord("q"):
            break
        
        # Pause the loop if 'p' is pressed
        if wkey == ord("p"):
            cv2.waitKey(0)

    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
