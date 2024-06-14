# obsolete

import os

dirname = os.path.dirname(__file__)
os.chdir(dirname)
datayaml=os.path.join(dirname,
                      "datasets", "Playing Cards.v3-original_raw-images.yolov8", "data.yaml"
)

if not os.path.isfile(datayaml):
    raise FileNotFoundError(datayaml)

import datetime
from ultralytics import YOLO

yolo=YOLO("yolov8n.yaml")
if __name__ == "__main__":
    print(datetime.datetime.now())
    yolo.train(task="detect",mode="train",data=datayaml,device=0,patience=2,verbose=True,epochs=100)
    print(datetime.datetime.now())
