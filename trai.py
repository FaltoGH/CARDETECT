import os

dirname = os.path.dirname(__file__)
os.chdir(dirname)
datayaml=os.path.join(dirname,"datasets", "realimages", "dataset.yaml")
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
    yolo.train(task="detect",mode="train",data=datayaml,device=0,patience=2,verbose=True,epochs=10)
    print(datetime.datetime.now())
