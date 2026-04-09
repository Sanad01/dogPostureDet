from os import listdir
from pathlib import Path
import cv2
import csv
import numpy as np
from ultralytics import YOLO
from extractFeatures import extract_features

CSV_PATH = Path("/data/classifier/kp_data.csv")
model = YOLO("models/poseEstimator/weights/best.pt")
count = 0

label_map = {
    "standing": 0,
    "sitting": 1,
    "lying": 2
}

SOURCE_IMAGE_ROOT = Path("/data/classifier/images")

DEST_BASE = Path("/data/classifier")

def normalize_keypoints(kp):
    kp = np.array(kp)[:, :2]

    # center (remove position)
    center = kp.mean(axis=0)
    kp = kp - center

    # scale (use nose-tail distance)
    nose = kp[16]
    tail = kp[12]
    scale = np.linalg.norm(nose - tail) + 1e-6

    kp = kp / scale

    return kp

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)

    for label_name in listdir(SOURCE_IMAGE_ROOT):
        folder = SOURCE_IMAGE_ROOT / label_name
        for image_name in listdir(folder):
            image_path = folder / image_name

            image = cv2.imread(str(image_path))
            if image is None:
                continue

            result = model(image)

            if result[0].keypoints is None or len(result[0].keypoints.xy) == 0:
                print(f"no keypoint in {image_name}")
                continue

            conf = result[0].keypoints.conf[0].cpu().numpy()
            if np.mean(conf) < 0.5: #skip low confidence key points
                continue

            kp = result[0].keypoints.xy[0].cpu().numpy() #only gets one dog per image also moves tensor to cpu

            try:
                kp_norm = normalize_keypoints(kp)
                features = extract_features(kp_norm)
                writer.writerow(list(features) + [label_map[label_name]])
                count += 1
            except Exception as e:
                print(f"Error processing {image_name}: {e}")
                continue

    print(f"Saved to {count} kp_data.csv")
