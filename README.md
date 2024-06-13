# YOLOv8-Playing-Card-Detect (incomplete)

## YOLOv8n
YOLOv8n train result:
![image](/runs/detect/train/results.png)

## Experiment results
- Missing box, additional box, or not matched label are counted as one wrong.
- Otherwise, it is counted as one correct.

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

- QC 9C X
- KC KC O
- 9D 9D O
- QD 9D X
- 10H 10H O
- 7H 7H O
- 6H 6H O
- AH AH O 
- 10S 10S O
- JS JC X
- 8S 8C X
- 3D 5D X
- 8H 6H X
- 5D 5D O
- 5H 5H O
- 9H_1 9H O
- 9H_2 7H X
- KD KD O
- KH_1 ? X
- KH_2 ? X
- 11 of 20 are correct if angle-invariant algorithm applied.

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

- QC QC O
- KC KC O 
- 9D 9D O
- QD 9D X
- 10H 10H O
- 7H 7H O
- 6H 4H X
- AH AH O
- 10S 10S O
- JS JS O
- 8S 8S O
- 3D 3D O
- 8H 8H O
- 5D 5D O
- 5H 5H O
- 9H_1 7H X
- 9H_2 9H O
- KD KD O
- KH_1 KH O
- KH_2 KH O
- 17 of 20 are correct if angle-invariant algorithm applied.

## Conclusion
YOLOv8s performance result is better than YOLOv8n's.

So I decided to use YOLOv8s.

## Rotational Invariance Prediction (RIP)
Rotational invariance algorithm is pretty simple.

Just rotate a image for 0, 90, 180, and 270 degree respectively.

And predict those images.

And merge those four results into one.

In merging process, two bounding boxes of IOU > 0.4 is considered as the same object.

Detail is implemented in `rotational_invariance_pred.py`, in about 480 lines.

Following is the prediction result.

![image](/images/0.jpg_r_pred.jpg)

## Confine
Algorithm `confine` is pretty simple.

If YOLOv8 has detected `7` but there is a confined space in the image, it is considered that YOLOv8 misdetected 9 as 7. The basic idea is that 7 has no circle but 9 has.

![image](/images/9H.png)

The algorithm to determine if there exists any confined space is implemented in `confine.py`, in about 180 lines.

## Rotational Invariance Prediction + Confine

### Result
![image](/images/0.jpg_rc_pred.jpg)

### Limitation
Confine algorithm cannot distinguish Q and 9, 4 and 6.
