import cv2
import numpy as np
from ultralytics import YOLO
import joblib
from extractFeatures import extract_features
from collections import deque

prev_kp = None
history = deque(maxlen=5)
alpha = 0.7

# load models
pose_estimator = YOLO("models/poseEstimator/weights/best.pt")
classifier = joblib.load("models/classifier/posture_model.pkl")
detector = YOLO("models/dogDetector/weights/best.pt")

label_map = {
    0: "Standing",
    1: "Sitting",
    2: "Lying"
}

def normalize_keypoints(kp):
    kp = np.array(kp)[:, :2] # first two columns of all rows

    if kp.shape[0] < 17:
        return None

    center = kp.mean(axis=0)
    kp = kp - center

    nose = kp[16]
    tail = kp[12]

    scale = np.linalg.norm(nose - tail)
    if scale < 1e-6:
        return None

    kp = kp / scale
    return kp

# webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    det_results = detector(frame)

    if det_results[0].boxes is not None and len(det_results[0].boxes) > 0: # will only detect on dog
        # get first detection (or best one)
        box = det_results[0].boxes.xyxy[0].cpu().numpy() #tensor
        x1, y1, x2, y2 = map(int, box)

        # crop dog region
        crop = frame[y1:y2, x1:x2]

        # run pose model on cropped image
        results = pose_estimator(crop)

        print(det_results[0].keypoints)

        if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0: # will only detect one dog

            kp = results[0].keypoints.xy[0].cpu().numpy()

            kp[:, 0] += x1
            kp[:, 1] += y1

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            if prev_kp is not None: # frame smoothing
                kp = alpha * prev_kp + (1 - alpha) * kp

            prev_kp = kp

            for x, y in kp:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

            conf = results[0].keypoints.conf[0].cpu().numpy()

            if len(kp) >= 17: #and np.mean(conf) > 0.5 and np.min(conf) > 0.3:

                kp_norm = normalize_keypoints(kp)

                if kp_norm is not None:
                    try:
                        features = extract_features(kp_norm)
                        features = np.array(features).reshape(1, -1)

                        pred = clf.predict(features)[0] # returns class index
                        history.append(pred)
                        pred = max(set(history), key=history.count)

                        label = label_map[int(pred)]

                        # draw label
                        cv2.putText(frame, label, (30, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1.5, (0, 255, 0), 3)

                    except:
                        pass

    else:
        cv2.putText(frame, "NO DETECTION", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Dog Posture Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
