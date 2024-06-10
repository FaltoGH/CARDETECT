# yolov8n-playing-card-object-detection

## YOLOv8n
YOLOv8n train result:
![image](/runs/detect/train/results.png)

## Experiment results
Missing box, additional box, or not matched label is counted as one wrong.

Predict result:
![image](/images/0.jpg_pred.jpg)
- Correct: 12
- Wrong: 9

Clockwise 90 result:
![image](/images/1.jpg_pred.jpg)
- Correct: 9
- Wrong: 12

Clockwise 180 result:
![image](/images/2.jpg_pred.jpg)
- Correct: 7
- Wrong: 14

Clockwise 270 result:
![image](/images/3.jpg_pred.jpg)
- Correct: 10
- Wrong: 11

## YOLOv8s
### [Experiment results](https://github.com/PD-Mera/Playing-Cards-Detection?tab=readme-ov-file#experiment-results)

With 10 epochs for each experiments

| Models | mAP50 | mAP50:95 | Size |
|:---:|:---:|:---:|:---:|
| [yolov8s](https://drive.google.com/file/d/1AqZnW6dI6flFZvGxAn6A9apDNSviXZ5f/view?usp=share_link) | 0.99498 | 0.95681 | 22.0MB |

Predict result:
![image](/images/0.jpg_s_pred.jpg)
- Correct: 17
- Wrong: 4

Clockwise 90 result:
![image](/images/1.jpg_s_pred.jpg)
- Correct: 16
- Wrong: 4

Clockwise 180 result:
![image](/images/2.jpg_s_pred.jpg)
- Correct: 16
- Wrong: 4

Clockwise 270 result:
![image](/images/3.jpg_s_pred.jpg)
- Correct: 14
- Wrong: 7

# Conclusion
YOLOv8s is better than YOLOv8n.
