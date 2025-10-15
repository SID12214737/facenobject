from deepface import DeepFace
from ultralytics import YOLO
import cv2

# Load YOLOv8n (nano, fastest model)
yolo = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for faster inference
    small_frame = cv2.resize(frame, (640, 480))

    # ---------- Emotion detection ----------
    text = ""
    try:
        result = DeepFace.analyze(small_frame, actions=['emotion'], enforce_detection=True)
        emotion = result[0]['dominant_emotion']
        text = f"Emotion: {emotion}"
    except:
        text = "No face or spoof detected"

    cv2.putText(small_frame, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # ---------- Object detection ----------
    results = yolo.predict(small_frame, conf=0.5, verbose=False)

    # Draw YOLO boxes
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{r.names[int(box.cls[0])]} {box.conf[0]:.2f}"
            cv2.rectangle(small_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(small_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # ---------- Display ----------
    cv2.imshow("Emotion + Object Detection", small_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
