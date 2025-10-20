# server_realsense_faces.py
import os
import time
import math
import json
import threading
import asyncio

import cv2
import numpy as np
import pyrealsense2 as rs

from aiohttp import web
from av import VideoFrame
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
import aiohttp_cors

# InsightFace
import insightface
from insightface.app import FaceAnalysis

# ---------------- Config ----------------
DISPLAY = True
MAX_FACES = 3
VIDEO_WIDTH = 1024
VIDEO_HEIGHT = 768
VIDEO_FPS = 30
# ----------------------------------------

# Globals shared between threads
_latest_frame = None            # numpy BGR image (annotated)
_latest_detections = []         # list of dicts (metadata)
_frame_lock = threading.Lock()
_running = True
_pipeline = None                # real sense pipeline object to stop later

# Initialize InsightFace
app = FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def load_known_faces(app, directory="faces"):
    known_faces = {}
    if not os.path.isdir(directory):
        print(f"[WARN] Known faces directory '{directory}' not found.")
        return known_faces

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

# ---------- pose estimation ----------
def estimate_head_pose(landmarks_2d, frame_width, frame_height):
    if landmarks_2d is None or len(landmarks_2d) < 5:
        return None, None, None, "No landmarks"

    # model points correspond roughly to the 5 keypoints we have
    model_points = np.array([
        [-30.0, 40.0, 30.0],   # left eye
        [30.0, 40.0, 30.0],    # right eye
        [0.0, 0.0, 0.0],       # nose tip
        [-25.0, -40.0, 30.0],  # left mouth corner
        [25.0, -40.0, 30.0]    # right mouth corner
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

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_EPNP
    )

    if not success:
        return None, None, None, "Pose not found"

    rot_mat, _ = cv2.Rodrigues(rotation_vector)

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
        description.append("Chapga")
    elif yaw < -15:
        description.append("Ongga")
    else:
        description.append("Tog'riga")

    if pitch > 150:
        description.append("Pastga")
    elif pitch < -150:
        description.append("Tepaga")
    else:
        description.append("Tog'riga")

    head_pose_desc = " & ".join(description)
    euler_angles = {"yaw": float(yaw), "pitch": float(pitch), "roll": float(roll)}

    return rotation_vector, translation_vector, euler_angles, head_pose_desc

# ---------- detection on a frame ----------
def detect_faces_and_objects(frame):
    """Run insightface on frame, annotate it, and return (annotated_frame, detection_info_list)."""
    detection_info = []
    frame_h, frame_w = frame.shape[:2]
    frame_center_x = frame_w / 2
    frame_center_y = frame_h / 2

    faces = app.get(frame)
    if faces is None:
        faces = []

    faces = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]), reverse=True)[:MAX_FACES]

    for f in faces:
        bbox = f.bbox.astype(int)
        x1 = int(max(0, bbox[0])); y1 = int(max(0, bbox[1]))
        x2 = int(min(frame_w - 1, bbox[2])); y2 = int(min(frame_h - 1, bbox[3]))

        entry = {
            "class": 'person',
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "extra": {}
        }

        # Recognize via embeddings
        name = "Unknown"
        max_sim = 0.0
        for known_name, known_emb in known_faces.items():
            sim = float(np.dot(f.embedding, known_emb) / (np.linalg.norm(f.embedding) * np.linalg.norm(known_emb)))
            if sim > 0.4 and sim > max_sim:
                max_sim = sim
                name = known_name

        # position (normalized by half-width/height of detection box)
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        dx = (center_x - frame_center_x) / ((x2 - x1) / 2 + 1e-6)
        dy = (center_y - frame_center_y) / ((y2 - y1) / 2 + 1e-6)

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

        # head pose
        head_pose = None
        if hasattr(f, "kps") and f.kps is not None and len(f.kps) >= 5:
            # f.kps is a (5,2) array. pass as-is.
            rot_vec, trans_vec, angles, desc = estimate_head_pose(f.kps, frame_w, frame_h)
            head_pose = {"angle-desc": desc, "angles": angles}
        else:
            head_pose = None

        entry["extra"].update({
            "recognized-name": name,
            "similarity": float(max_sim),
            "position-desc": position_desc,
            "head-pose": head_pose
        })

        # annotate on frame
        if DISPLAY:
            cv2.putText(frame, f"{name}", (x1, max(10, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            for (x, y) in getattr(f, "kps", []):
                cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            if head_pose is not None:
                cv2.putText(frame, head_pose["angle-desc"], (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

        detection_info.append(entry)

    return frame, detection_info

# ---------- RealSense capture thread ----------
def realsense_capture_loop(realsense=False):
    global _latest_frame, _latest_detections, _running, _pipeline
    # Configure depth and color streams
    if realsense:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, VIDEO_WIDTH, VIDEO_HEIGHT, rs.format.bgr8, VIDEO_FPS)

        profile = pipeline.start(config)
        _pipeline = pipeline
        print("[INFO] RealSense pipeline started.")
    else:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print("[INFO] USB camera pipeline started.")


    try:
        while _running:
            if realsense:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                color_image = np.asanyarray(color_frame.get_data())
            else:
                for _ in range(0):
                    cap.grab()

                ret, color_image = cap.read()
                if not ret:
                    break
            
            
            # run detection & annotation
            annotated, detections = detect_faces_and_objects(color_image)

            with _frame_lock:
                _latest_frame = annotated.copy()
                # store a serializable version of detections
                _latest_detections = detections

            if DISPLAY:
                cv2.putText(annotated, f"faces: {len(detections)}", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # cv2.imshow("RealSense - annotated", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[INFO] 'q' pressed - stopping.")
                    _running = False
                    break

    except Exception as e:
        print(f"[ERROR] RealSense capture loop exception: {e}")
    finally:
        try:
            pipeline.stop()
            print("[INFO] RealSense pipeline stopped.")
        except Exception:
            pass
        _pipeline = None

# Start capture thread
_capture_thread = threading.Thread(target=realsense_capture_loop, daemon=True)
_capture_thread.start()

# ---------- WebRTC broadcast track ----------
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
            frame = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)

        # convert to av.VideoFrame (bgr24)
        video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        return video_frame

broadcast_track = BroadcastTrack(fps=VIDEO_FPS)
pcs = set()

# ---------- HTTP / WebRTC handlers ----------
BASE_DIR = os.path.dirname(__file__)

async def index(request):
    path = os.path.join(BASE_DIR, "index.html")
    if not os.path.exists(path):
        return web.Response(status=404, text="index.html not found - please provide a client page.")
    html = open(path, "r").read()
    return web.Response(content_type="text/html", text=html)

async def offer(request):
    try:
        # Try to parse JSON from the request body
        params = await request.json()
    except json.JSONDecodeError:
        # If the request had no JSON (e.g., empty body)
        return web.Response(text="Invalid or empty JSON payload", status=400)

    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)
    print("[INFO] New PeerConnection:", pc)

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

async def detections_handler(request):
    """Return the latest detections as JSON."""
    with _frame_lock:
        data = _latest_detections.copy() if _latest_detections is not None else []
    return web.Response(content_type="application/json", text=json.dumps(data))

async def on_shutdown(app):
    global _running, _pipeline
    print("[INFO] Shutting down server...")
    _running = False
    # stop Realsense pipeline if still running
    try:
        if _pipeline is not None:
            _pipeline.stop()
    except Exception:
        pass
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

# ---------- MJPEG video stream ----------
async def mjpeg_handler(request):
    """Serve MJPEG video stream."""
    boundary = "frame"
    response = web.StreamResponse(
        status=200,
        reason="OK",
        headers={
            "Content-Type": f"multipart/x-mixed-replace; boundary={boundary}"
        },
    )
    await response.prepare(request)

    try:
        while _running:
            with _frame_lock:
                frame = _latest_frame.copy() if _latest_frame is not None else None

            if frame is None:
                frame = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)

            _, jpeg = cv2.imencode(".jpg", frame)
            data = jpeg.tobytes()

            await response.write(
                b"--" + boundary.encode() + b"\r\n"
                + b"Content-Type: image/jpeg\r\n"
                + f"Content-Length: {len(data)}\r\n\r\n".encode()
                + data
                + b"\r\n"
            )

            await asyncio.sleep(1.0 / VIDEO_FPS)
    except asyncio.CancelledError:
        print("[INFO] MJPEG stream cancelled.")
    except Exception as e:
        print(f"[ERROR] MJPEG stream: {e}")
    finally:
        try:
            await response.write_eof()
        except Exception:
            pass
    return response

# ---------- main server ----------
if __name__ == "__main__":
    webapp = web.Application()

    cors = aiohttp_cors.setup(webapp, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods="*",
        )
    })

    webapp.on_shutdown.append(on_shutdown)

    resource = cors.add(webapp.router.add_resource("/offer"))
    cors.add(resource.add_route("POST", offer))

    resource = cors.add(webapp.router.add_resource("/video"))
    cors.add(resource.add_route("GET", mjpeg_handler))

    resource = cors.add(webapp.router.add_resource("/"))
    cors.add(resource.add_route("GET", index))

    resource = cors.add(webapp.router.add_resource("/detections"))
    cors.add(resource.add_route("GET", detections_handler))
    
    print("[INFO] Starting WebRTC server on 0.0.0.0:8088")
    web.run_app(webapp, host="0.0.0.0", port=8088)
