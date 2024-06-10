import os

dirname = os.path.dirname(__file__)
os.chdir(dirname)
model=os.path.join(dirname, "yolov8s_playing_cards.pt")
test=os.path.join(dirname, "images")

if not os.path.isfile(model):
    raise FileNotFoundError(model)

if not os.path.isdir(test):
    raise FileNotFoundError(test)

WINNAME = "YOLOv8 Inference"

import cv2
from ultralytics import YOLO


# Load the YOLOv8 model
model = YOLO(model)

# Run YOLOv8 inference on the frame
for im in os.listdir(test):
    if im.endswith("_pred.jpg"): continue

    abspath = os.path.join(test, im)
    result = model(abspath)[0]
    result.save(result.path + "_s_pred.jpg")

    # Visualize the results on the frame
    annotated_frame = result.plot()

    width = annotated_frame.shape[1]
    height = annotated_frame.shape[0]
    print("width=%d, height=%d"%(width, height))

    while width > 1800 or height > 900:
        annotated_frame = cv2.resize(annotated_frame, (width>>1, height>>1))
        width = annotated_frame.shape[1]
        height = annotated_frame.shape[0]
    
    # Display the annotated frame
    cv2.imshow(WINNAME, annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(0) & 0xFF == ord("q"):
        break

# Close the display window
cv2.destroyAllWindows()
