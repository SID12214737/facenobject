import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import time
import os
import math

import pyrealsense2 as rs


DISPLAY = True
MAX_FACES = 3

app = FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def load_known_faces(app, directory="faces"):
    known_faces = {}

    for filename in os.listdir(directory):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            name = os.path.splitext(filename)[0]  
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

    detection_info = []

    frame_h, frame_w = frame.shape[:2]
    frame_center_x = frame_w / 2
    frame_center_y = frame_h / 2

    faces = app.get(frame)

    faces = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]), reverse=True)[:MAX_FACES]

    for f in faces:
        bbox = f.bbox.astype(int)
        x1 = int(max(0, bbox[0]))
        y1 = int(max(0, bbox[1]))
        x2 = int(min(frame_w - 1, bbox[2]))
        y2 = int(min(frame_h - 1, bbox[3]))

        entry = {
                "class": 'person',
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
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

def send_frame():
    
    return

def main():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("Color sensor not found")
        exit(0)

    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    prev_frame_time = time.time()
    new_frame_time = 0

    try:
        while True:
            new_frame_time = time.time()
            fps = round(1 / (new_frame_time - prev_frame_time))
            prev_frame_time = new_frame_time

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())

            images, detections = detect_faces_and_objects(color_image)
            
            print(f"[FPS]{fps}")
            print(f"[DETECTIONS] {detections}\n")

            if DISPLAY:
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.putText(images, f"fps: {fps}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                cv2.imshow('RealSense', images)
            cv2.waitKey(1)

    finally:

        # Stop streaming
        pipeline.stop()

# server.py
import asyncio
import threading
import time
import cv2
import os
import numpy as np
from aiohttp import web
from av import VideoFrame
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack

BASE_DIR = os.path.dirname(__file__)
VIDEO_HEIGHT = 640
VIDEO_WIDTH = 480
VIDEO_FPS = 60

cap = cv2.VideoCapture(0)  
cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, VIDEO_FPS)

_latest_frame = None
_frame_lock = threading.Lock()
_running = True

def capture_loop():
    global _latest_frame, _running
    while _running:
        ret, frame = cap.read()
        if ret:
            with _frame_lock:
                _latest_frame = frame.copy()
        else:
            time.sleep(0.01)

t = threading.Thread(target=capture_loop, daemon=True)
t.start()

class BroadcastTrack(VideoStreamTrack):
    
    def __init__(self, fps=VIDEO_FPS):
        super().__init__()
        self._fps = fps
        self._frame_time = 1.0 / fps
        self._last_ts = None

    async def recv(self):
        if self._last_ts is None:
            self._last_ts = time.time()
        else:
            elapsed = time.time() - self._last_ts
            if elapsed < self._frame_time:
                await asyncio.sleep(self._frame_time - elapsed)
            self._last_ts = time.time()

        with _frame_lock:
            frame = _latest_frame.copy() if _latest_frame is not None else None

        if frame is None:
            # create a black frame if nothing is captured yet
            frame = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)

        # Create an av.VideoFrame from numpy (OpenCV uses BGR)
        video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        return video_frame

broadcast_track = BroadcastTrack(fps=30)

pcs = set()

async def index(request):
    html = open(os.path.join(BASE_DIR, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=html)

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)
    print("New PeerConnection:", pc)

    pc.addTrack(broadcast_track)

    @pc.on("connectionstatechange")
    async def on_state_change():
        print("Connection state:", pc.connectionState)
        if pc.connectionState in ("failed", "closed"):
            await pc.close()
            pcs.discard(pc)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

async def on_shutdown(app):
    global _running
    _running = False
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()
    try:
        cap.release()
    except Exception:
        pass

if __name__ == "__main__":
    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)
    web.run_app(app, host="0.0.0.0", port=8088)
