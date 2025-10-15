import cv2
import numpy as np
from ultralytics.models import YOLO
import insightface
from insightface.app import FaceAnalysis

# --- Initialize YOLO (for objects and people)
yolo = YOLO("yolov8n.pt")  # lightweight model, use yolov8m or l for more accuracy

# --- Initialize InsightFace
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# --- Build face embeddings database ---
known_faces = {}
known_images = {
    "Mir": "faces/mir.jpg",
}

for name, path in known_images.items():
    img = cv2.imread(path)
    faces = app.get(img)
    if faces:
        known_faces[name] = faces[0].embedding

print(f"[INFO] Loaded {len(known_faces)} known faces.")

def detect_blink(landmarks, prev_state=None):
    # Using eye aspect ratio (EAR)
    def eye_ratio(p):
        A = np.linalg.norm(p[1] - p[5])
        B = np.linalg.norm(p[2] - p[4])
        C = np.linalg.norm(p[0] - p[3])
        return (A + B) / (2.0 * C)

    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]
    ear = (eye_ratio(left_eye) + eye_ratio(right_eye)) / 2.0
    return 1.0 if ear < 0.2 else 0.0


def texture_liveness(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]

    # Clamp coordinates to frame bounds
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    # Skip invalid boxes
    if x2 <= x1 or y2 <= y1:
        return False

    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return False

    # Convert to grayscale and compute texture variance
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Adjust threshold based on experiments; ~100–150 is a good start
    return lap_var > 120

def color_liveness(face):
    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
    mean_hue = np.mean(hsv[..., 0])
    mean_sat = np.mean(hsv[..., 1])
    return mean_sat > 40 and (10 < mean_hue < 170)


def liveness_check(frame, face, prev_state=None):
    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = np.clip(bbox, 0, [frame.shape[1]-1, frame.shape[0]-1, frame.shape[1]-1, frame.shape[0]-1])
    if x2 <= x1 or y2 <= y1:
        return 0.0

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0

    # --- Texture cue ---
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    texture_score = np.tanh((lap_var - 80) / 60) * 0.5 + 0.5   # map to 0-1

    # --- Color cue ---
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mean_sat = np.mean(hsv[..., 1])
    mean_val = np.mean(hsv[..., 2])
    color_score = np.clip((mean_sat - 30) / 50, 0, 1) * np.clip((mean_val - 80) / 80, 0, 1)

    # --- Blink / micro-motion cue ---
    blink_score = 0.0
    if hasattr(face, "landmark_3d_68") and face.landmark_3d_68 is not None:
        blink_score = detect_blink(face.landmark_3d_68[:, :2], prev_state)

    # Weighted fusion
    weights = np.array([0.5, 0.3, 0.2])
    scores = np.array([texture_score, color_score, blink_score])
    confidence = float(np.dot(weights, scores))

    return confidence   # 0–1 float instead of bool


def detect_faces_and_objects(frame):

    # Step 1: Run YOLO detection
    results = yolo(frame, verbose=False)
    detections = results[0].boxes.data.cpu().numpy()

    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        cls_name = yolo.names[int(cls_id)]

        if cls_name in ["person", "face"]:
            # Crop the face/person region
            crop = frame[int(y1):int(y2), int(x1):int(x2)]
            if crop.size == 0:
                continue

            # Step 2: Run InsightFace recognition + liveness
            faces = app.get(crop)
            for f in faces:
                # Compute similarity with known faces
                name = "Unknown"
                max_sim = 0
                for known_name, known_emb in known_faces.items():
                    sim = np.dot(f.embedding, known_emb) / (
                        np.linalg.norm(f.embedding) * np.linalg.norm(known_emb)
                    )
                    if sim > 0.4 and sim > max_sim:  # threshold for match
                        max_sim = sim
                        name = known_name

                # Simple liveness estimation
                liveness_conf = liveness_check(frame, f, prev_state=None)
                is_live = liveness_conf > 0.6
                color = (0, 255, 0) if is_live else (0, 0, 255)

                # Draw bounding box & label
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"{name} {'| live:'} {liveness_conf:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        else:
            # For other YOLO objects (car, cup, etc.)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
            cv2.putText(frame, cls_name, (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    return frame

def main():
    # --- Open webcam ---
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while True:
        for _ in range(3):  # discard buffered frames
            cap.grab()

        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_faces_and_objects(frame)

        cv2.imshow("Face & Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
