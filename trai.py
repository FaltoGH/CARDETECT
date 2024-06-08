import os

dirname = os.path.dirname(__file__)
os.chdir(dirname)
datayaml=os.path.join(dirname,
                      "datasets", "Playing Cards.v3-original_raw-images.yolov8", "data.yaml"
)
model=os.path.join(dirname,"runs","detect","train7","weights","best.pt")

if not os.path.isfile(datayaml):
    raise FileNotFoundError(datayaml)

if not os.path.isfile(model):
    raise FileNotFoundError(model)

import datetime
from ultralytics import YOLO

yolo=YOLO(model)
if __name__ == "__main__":
    print(datetime.datetime.now())
    yolo.train(
task="detect",
mode="train",
data=datayaml,
device=0,
patience=2,
verbose=True,
epochs=100,
batch=16,
imgsz=720,
save=True,
save_period=-1,
cache=False,
workers=1,
exist_ok=False,
pretrained=True,
seed=0,
deterministic=True,
single_cls=False,
dropout=0.1,
val=True,
save_json=False,
save_hybrid=False,
augment=False,
show=False,
save_frames=False,
save_txt=False,
save_conf=False,
save_crop=False,
show_labels=True,
show_conf=True,
show_boxes=True,
)
    print(datetime.datetime.now())
