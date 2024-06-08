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
project=None,
exist_ok=False,
pretrained=True,
optimizer="auto",
seed=0,
deterministic=True,
single_cls=False,
dropout=0.1,
val=True,
split="val",
save_json=False,
save_hybrid=False,
conf=None,
iou=0.7,
max_det=300,
half=False,
dnn=False,
plots=True,
source=None,
vid_stride=1,
stream_buffer=False,
visualize=False,
augment=False,
agnostic_nms=False,
classes=None,
retina_masks=False,
embed=None,
show=False,
save_frames=False,
save_txt=False,
save_conf=False,
save_crop=False,
show_labels=True,
show_conf=True,
show_boxes=True,
line_width=None,
format="torchscript",
keras=False,
optimize=False,
int8=False,
dynamic=False,
simplify=False,
opset=None,
workspace=4,
nms=False,
box=7.5,
cls=0.5,
dfl=1.5,
pose=12.0,
kobj=1.0,
label_smoothing=0.0,
nbs=64,
hsv_h=0.015,
hsv_s=0.7,
hsv_v=0.4,
degrees=0.0,
translate=0.1,
scale=0.5,
shear=0.0,
perspective=0.0,
flipud=0.0,
fliplr=0.5,
bgr=0.0,
mosaic=1.0,
mixup=0.0,
copy_paste=0.0,
auto_augment="randaugment",
erasing=0.4,
crop_fraction=1.0,
cfg=None,
tracker="botsort.yaml"
)
    print(datetime.datetime.now())
