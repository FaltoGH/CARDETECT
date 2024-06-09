import os

dirname = os.path.dirname(__file__)
os.chdir(dirname)
model=os.path.join(dirname, "runs", "detect", "train", "weights", "best.pt")
test=os.path.join(dirname, "images")

if not os.path.isfile(model):
    raise FileNotFoundError(model)
if not os.path.isdir(test):
    raise FileNotFoundError(test)

from ultralytics import YOLO
import cv2

# Load the YOLOv8 model
model = YOLO(model)

# Run YOLOv8 inference on the frame
for im in os.listdir(test):
    if im.endswith("_pred.jpg"): continue

    abspath = os.path.join(test, im)
    result = model(abspath)[0]
    result.save(result.path + "_pred.jpg")

    # Visualize the results on the frame
    annotated_frame = result.plot()

    # Display the annotated frame
    cv2.imshow("YOLOv8 Inference", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(0) & 0xFF == ord("q"):
        break

# Close the display window
cv2.destroyAllWindows()
