import cv2
import numpy as np
from ultralytics.models import YOLO
import insightface
from insightface.app import FaceAnalysis
import time
import os
import math
import threading

from onnxruntime.quantization import quantize_dynamic, QuantType

# --- Initialize YOLO (for objects and people)
yolo = YOLO("yolov5nu.pt")
yolo.export(format="onnx", opset=12)


# --- Initialize InsightFace
app = FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(320, 320))

# --- Build face embeddings database ---
def load_known_faces(app, directory="faces"):
    known_faces = {}

    # List all image files in directory
    for filename in os.listdir(directory):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            name = os.path.splitext(filename)[0]  # get filename without extension
            path = os.path.join(directory, filename)
            img = cv2.imread(path)

            if img is None:
                print(f"[WARN] Failed to read image: {path}")
                continue

            faces = app.get(img)
            if faces:
                known_faces[name] = faces[0].embedding
                print(f"[INFO] Loaded face for {name}")
            else:
                print(f"[WARN] No face detected in {filename}")

    print(f"[INFO] Loaded {len(known_faces)} known faces total.")
    return known_faces

known_faces = load_known_faces(app, "faces")

def register_new_person():
    return

def recognize_emotion():
    return

def estimate_head_pose(landmarks_2d, frame_width, frame_height):
    if landmarks_2d is None or len(landmarks_2d) < 5:
        return None, None, None, "No landmarks"

    model_points = np.array([
        [-30.0, 40.0, 30.0],   # Left eye
        [30.0, 40.0, 30.0],    # Right eye
        [0.0, 0.0, 0.0],       # Nose tip
        [-25.0, -40.0, 30.0],  # Left mouth corner
        [25.0, -40.0, 30.0]    # Right mouth corner
    ], dtype=np.float64)

    image_points = np.array(landmarks_2d, dtype=np.float64).reshape(-1, 2)

    # Camera matrix
    focal_length = frame_width
    center = (frame_width / 2, frame_height / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # Pose estimation
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_EPNP
    )

    if not success:
        return None, None, None, "Pose not found"

    # Convert rotation vector to rotation matrix
    rot_mat, _ = cv2.Rodrigues(rotation_vector)

    # Extract Euler angles (in degrees)
    sy = math.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = math.degrees(math.atan2(rot_mat[2, 1], rot_mat[2, 2]))
        yaw = math.degrees(math.atan2(-rot_mat[2, 0], sy))
        roll = math.degrees(math.atan2(rot_mat[1, 0], rot_mat[0, 0]))
    else:
        pitch = math.degrees(math.atan2(-rot_mat[1, 2], rot_mat[1, 1]))
        yaw = math.degrees(math.atan2(-rot_mat[2, 0], sy))
        roll = 0

    description = []
    if yaw > 15:
        description.append("Looking Left")
    elif yaw < -15:
        description.append("Looking Right")
    else:
        description.append("Facing Forward")

    if pitch > 10:
        description.append("Looking Down")
    elif pitch < -10:
        description.append("Looking Up")

    head_pose_desc = " & ".join(description)

    euler_angles = {"yaw": yaw, "pitch": pitch, "roll": roll}

    return rotation_vector, translation_vector, euler_angles, head_pose_desc

def detect_faces_and_objects(frame):
    # Step 1: Run YOLO detection
    results = yolo(frame, verbose=False)
    detections = results[0].boxes.data.cpu().numpy()

    # Detection summary list
    detection_info = []

    frame_h, frame_w = frame.shape[:2]
    frame_center_x = frame_w / 2
    frame_center_y = frame_h / 2

    faces = app.get(frame)

    for f in faces:
        bbox = f.bbox.astype(int)
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]
        entry = {
                "class": 'person',
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                # "confidence": float(f.confidence),
                "extra": {}
            }
        # Compute similarity with known faces
        name = "Unknown"
        max_sim = 0
        for known_name, known_emb in known_faces.items():
            sim = np.dot(f.embedding, known_emb) / (
                np.linalg.norm(f.embedding) * np.linalg.norm(known_emb)
            )
            if sim > 0.4 and sim > max_sim:
                max_sim = sim
                name = known_name
        if DISPLAY:
            cv2.putText(frame, name, (int(x1), int(y1) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Compute center of the detected person/face
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Compute relative center offset (normalized)
        dx = (center_x - frame_center_x) / ((x2 - x1) / 2)
        dy = (center_y - frame_center_y) / ((y2 - y1) / 2)

        horizontal = "center"
        vertical = "center"

        if dx < -1:
            horizontal = "left"
        elif dx > 1:
            horizontal = "right"

        if dy < -1:
            vertical = "top"
        elif dy > 1:
            vertical = "bottom"

        position_desc = f"{vertical}-{horizontal}"

        if hasattr(f, "kps"):
            rot_vec, trans_vec, angles, desc = estimate_head_pose(f.kps, frame.shape[1], frame.shape[0])
            head_pose = {
                "angle-desc": desc,
                "angles": angles,
                # "rotation_vector": rot_vec.tolist(),
                # "translation_vector": trans_vec.tolist()
            }
        else:
            head_pose = None

        # Add to detection metadata
        entry["extra"].update({
            "recognized-name": name,
            "similarity": float(max_sim),
            "position-desc": position_desc,
            "head-pose": head_pose,
        })
            
        if DISPLAY:
            for (x, y) in f.kps:
                cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)

        detection_info.append(entry)

    return frame, detection_info


# Set to True if wanna display output img
DISPLAY = True

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    prev_frame_time = time.time()
    new_frame_time = 0

    while True:

        for _ in range(0):
            cap.grab()

        ret, frame = cap.read()
        if not ret:
            break

        frame, detections = detect_faces_and_objects(frame)

        new_frame_time = time.time()
        fps = round(1 / (new_frame_time - prev_frame_time))
        prev_frame_time = new_frame_time

        ###
        print(f"[FPS]{fps}")
        print(f"[DETECTIONS] {detections}\n")

        if DISPLAY:
            cv2.putText(frame, f"fps: {fps}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            for det in detections:
                if det["class"] == "person" and det['extra']:
                    cv2.putText(frame, f"head-position: {det['extra']['position-desc']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 0), 2)
                    cv2.putText(frame, f"head-pose: {det['extra']['head-pose']['angle-desc']}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 100), 2)
                     
            cv2.imshow("Face & Object Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
