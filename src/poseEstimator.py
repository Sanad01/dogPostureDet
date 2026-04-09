import cv2
from ultralytics import YOLO

model = YOLO("models/poseEstimator/weights/best.pt")

kpt_names = [
"front_left_paw",
"front_left_knee",
"front_left_elbow",
"rear_left_paw",
"rear_left_knee",
"rear_left_elbow",
"front_right_paw",
"front_right_knee",
"front_right_elbow",
"rear_right_paw",
"rear_right_knee",
"rear_right_elbow",
"tail_start",
"tail_end",
"left_ear_base",
"right_ear_base",
"nose",
"chin",
"left_ear_tip",
"right_ear_tip",
"left_eye",
"right_eye",
"withers",
"throat"
]

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam")
    exit()

cv2.namedWindow("Dog Pose", cv2.WINDOW_NORMAL)


while True:

    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    annotated = frame.copy()

    for r in results:

        if r.keypoints is None or len(r.keypoints.xy) == 0:
            continue

        annotated = r.plot()

        keypoints = r.keypoints.xy[0].cpu().numpy()

        for i, (x, y) in enumerate(keypoints):
            name = kpt_names[i]

            cv2.circle(annotated, (int(x), int(y)), 5, (0, 255, 0), -1)

            cv2.putText(
                annotated,
                f"{i}",
                (int(x) + 5, int(y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1
            )


    # ALWAYS display frame
    cv2.imshow("Dog Pose", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
