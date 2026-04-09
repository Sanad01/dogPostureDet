import cv2
from ultralytics import YOLO

model = YOLO("models/dogDetector/weights/best.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam")
    exit()

cv2.namedWindow("Dog Detection", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    annotated = frame.copy()

    for r in results:

        if r.boxes is None or len(r.boxes) == 0:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()   # [x1, y1, x2, y2]
        confs = r.boxes.conf.cpu().numpy()   # confidence scores
        classes = r.boxes.cls.cpu().numpy()  # class ids

        for box, conf, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = map(int, box)

            # draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # label (you can customize this)
            label = f"Dog {conf:.2f}"

            cv2.putText(
                annotated,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # ALWAYS display frame
    cv2.imshow("Dog Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
