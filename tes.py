import os

dirname = os.path.dirname(__file__)
os.chdir(dirname)
model=os.path.join(dirname, "runs", "detect", "thirdtrain", "weights", "best.pt")
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
results = model(test, stream=True)

for result in results:
    # Visualize the results on the frame
    annotated_frame = result.plot()

    # Display the annotated frame
    cv2.imshow("YOLOv8 Inference", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(-1) & 0xFF == ord("q"):
        break

# Close the display window
cv2.destroyAllWindows()
