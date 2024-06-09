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
save=True,
save_period=-1,
cache=False,
workers=1,
exist_ok=False,
pretrained=True,
seed=0,
deterministic=True,
single_cls=False,
dropout=0.09,
val=True,
save_json=True,
save_hybrid=True,
augment=False,
show=False,
save_frames=True,
save_txt=True,
save_conf=True,
save_crop=True,
show_labels=True,
show_conf=True,
show_boxes=True,
)
    print(datetime.datetime.now())
