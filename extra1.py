#!/usr/bin/env python3
import os
import time
import math
import json
import threading
import asyncio

# COCO class name translations (Uzbek)
COCO_UZBEK = {
    "person": "odam",
    "bicycle": "velosiped",
    "car": "mashina",
    "motorcycle": "mototsikl",
    "airplane": "samolyot",
    "bus": "avtobus",
    "train": "poyezd",
    "truck": "yuk mashinasi",
    "boat": "qayiq",
    "traffic light": "svetofor",
    "fire hydrant": "gidrant",
    "stop sign": "to‘xtash belgisi",
    "parking meter": "to‘lov hisoblagich",
    "bench": "skameyka",
    "bird": "qush",
    "cat": "mushuk",
    "dog": "it",
    "horse": "ot",
    "sheep": "qo‘y",
    "cow": "sigir",
    "elephant": "fil",
    "bear": "ayiq",
    "zebra": "zebra",
    "giraffe": "jirafa",
    "backpack": "ryukzak",
    "umbrella": "soyabon",
    "handbag": "sumka",
    "tie": "galstuk",
    "suitcase": "chemodan",
    "frisbee": "disk",
    "skis": "chang‘i",
    "snowboard": "snoubord",
    "sports ball": "sport to‘pi",
    "kite": "laylak",
    "baseball bat": "beysboll tayoqchasi",
    "baseball glove": "beysboll qo‘lqopi",
    "skateboard": "skeytbord",
    "surfboard": "serfbord",
    "tennis racket": "tennis raketkasi",
    "bottle": "shisha",
    "wine glass": "vino bokali",
    "cup": "stakan",
    "fork": "vilka",
    "knife": "pichoq",
    "spoon": "qoshiq",
    "bowl": "idish",
    "banana": "banan",
    "apple": "olma",
    "sandwich": "sendvich",
    "orange": "apelsin",
    "broccoli": "brokkoli",
    "carrot": "sabzi",
    "hot dog": "xot-dog",
    "pizza": "pitsa",
    "donut": "ponchik",
    "cake": "tort",
    "chair": "stul",
    "couch": "divan",
    "potted plant": "guldon",
    "bed": "karavot",
    "dining table": "ovqat stoli",
    "toilet": "unitaz",
    "tv": "televizor",
    "laptop": "noutbuk",
    "mouse": "sichqoncha",
    "remote": "pult",
    "keyboard": "klaviatura",
    "cell phone": "telefon",
    "microwave": "mikroto‘lqinli pech",
    "oven": "pech",
    "toaster": "toster",
    "sink": "rakovina",
    "refrigerator": "muzlatkich",
    "book": "kitob",
    "clock": "soat",
    "vase": "vaza",
    "scissors": "qaychi",
    "teddy bear": "o‘yinchoq ayiq",
    "hair drier": "soch quritgich",
    "toothbrush": "tish cho‘tkasi"
}


# Silence Albumentations auto-update nag BEFORE anything may import it
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

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

from ultralytics.models import YOLO
import onnxruntime as ort

# ---------------- Headless / Config ----------------
def _is_headless():
    return not any(os.environ.get(var) for var in ("DISPLAY", "WAYLAND_DISPLAY", "MIR_SOCKET"))

HEADLESS = _is_headless()
DISPLAY = True if HEADLESS else True 
MAX_FACES = 5
MAX_OBJECTS = 5
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
VIDEO_FPS = 30
PROVIDERS = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]

# ----------------------------------------

# Globals shared between threads
_latest_frame = None            # numpy BGR image (annotated)
_latest_detections = []         # list of dicts (metadata)
_frame_lock = threading.Lock()
_running = True
_pipeline = None                # RealSense pipeline object to stop later

yolo = YOLO("yolov8n.pt")
yolo.export(format="onnx", opset=12)

# ---------- InsightFace init ----------
app = FaceAnalysis(name='buffalo_sc', providers=PROVIDERS)
# det_size gives better accuracy vs default
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

# --------- emotion prediction ---------
sess_options = ort.SessionOptions()
# emotion-deection model download link
# https://huggingface.co/webml/models-moved/resolve/0e73dc31942fbdbbd135d85be5e5321eee88a826/emotion-ferplus-8.onnx?download=true
emotion_sess = ort.InferenceSession("emotion-ferplus-8.onnx", providers=PROVIDERS)
input_name = emotion_sess.get_inputs()[0].name
output_name = emotion_sess.get_outputs()[0].name
emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
emotions = [
    "neytral",      # neutral  
    "xursand",      # happiness  
    "hayrat",       # surprise  
    "xafa",         # sadness  
    "g'azab",       # anger  
    "jirkanch",     # disgust  
    "qo'rquv",      # fear  
    "mensimaslik"   # contempt
]

def preprocess_face(img, size=(64, 64)):
    # FER+ expects grayscale [0,255], mean-centered around 128, no scaling to [0,1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, size)
    img = img.astype(np.float32)
    img = img - 128.0  # mean centering
    img = np.expand_dims(np.expand_dims(img, 0), 0)  # shape (1,1,64,64)
    return img

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=1, keepdims=True)

def predict_emotion(face_img):
    if face_img is None or not isinstance(face_img, np.ndarray) or face_img.size == 0:
        return "unknown"

    img = preprocess_face(face_img)
    preds = emotion_sess.run([output_name], {input_name: img})[0]
    probs = softmax(preds)
    emotion_id = int(np.argmax(probs))
    return emotions[emotion_id]

# ---------- pose estimation ----------
def estimate_head_pose(landmarks_2d, frame_width, frame_height):
    if landmarks_2d is None or len(landmarks_2d) < 5:
        return None, None, None, "No landmarks"

    model_points = np.array([
        [-30.0, 40.0, 30.0],   # left eye
        [30.0, 40.0, 30.0],    # right eye
        [0.0, 0.0, 0.0],       # nose tip
        [-25.0, -40.0, 30.0],  # left mouth corner
        [25.0, -40.0, 30.0]    # right mouth corner
    ], dtype=np.float64)

    image_points = np.array(landmarks_2d, dtype=np.float64).reshape(-1, 2)

    focal_length = frame_width
    center = (frame_width / 2, frame_height / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP
    )
    if not success:
        return None, None, None, "Pose not found"

    rot_mat, _ = cv2.Rodrigues(rotation_vector)
    sy = math.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = math.degrees(math.atan2(rot_mat[2, 1], rot_mat[2, 2]))
        yaw   = math.degrees(math.atan2(-rot_mat[2, 0], sy))
        roll  = math.degrees(math.atan2(rot_mat[1, 0], rot_mat[0, 0]))
    else:
        pitch = math.degrees(math.atan2(-rot_mat[1, 2], rot_mat[1, 1]))
        yaw   = math.degrees(math.atan2(-rot_mat[2, 0], sy))
        roll  = 0

    description = []
    if yaw > 15:
        description.append("Chapga")
    elif yaw < -15:
        description.append("Ongga")
    else:
        description.append("Tog'riga")

    if pitch > 15:
        description.append("Pastga")
    elif pitch < -15:
        description.append("Tepaga")
    else:
        description.append("Tog'riga")

    head_pose_desc = " & ".join(description)
    euler_angles = {"yaw": float(yaw), "pitch": float(pitch), "roll": float(roll)}
    return rotation_vector, translation_vector, euler_angles, head_pose_desc

# ---------- detection on a frame ----------
def detect_faces_and_objects(frame):
    results = yolo(frame, imgsz=480, verbose=False)
    detections = results[0].boxes.data.cpu().numpy()
    detection_info = []
    frame_h, frame_w = frame.shape[:2]
    frame_center_x = frame_w / 2
    frame_center_y = frame_h / 2

    # --- Step 1: Run InsightFace on the full frame once
    faces = app.get(frame)
    faces = sorted(
        faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]),
        reverse=True
    )[:MAX_FACES]

    for f in faces:
        bbox = f.bbox.astype(int)
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(frame_w - 1, x2); y2 = min(frame_h - 1, y2)

        name = "Aniqlanmagan"
        max_sim = 0.0
        if getattr(f, "embedding", None) is not None:
            for known_name, known_emb in known_faces.items():
                sim = float(np.dot(f.embedding, known_emb) /
                            (np.linalg.norm(f.embedding) * np.linalg.norm(known_emb)))
                if sim > 0.4 and sim > max_sim:
                    max_sim = sim
                    name = known_name

        # position in frame
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        dx = (center_x - frame_center_x) / ((x2 - x1) / 2 + 1e-6)
        dy = (center_y - frame_center_y) / ((y2 - y1) / 2 + 1e-6)

        horizontal = "center"
        vertical = "center"
        if dx < -1: horizontal = "left"
        elif dx > 1: horizontal = "right"
        if dy < -1: vertical = "top"
        elif dy > 1: vertical = "bottom"
        position_desc = f"{vertical}-{horizontal}"

        # head pose estimation
        head_pose = None
        if hasattr(f, "kps") and f.kps is not None and len(f.kps) >= 5:
            rot_vec, trans_vec, angles, desc = estimate_head_pose(f.kps, frame_w, frame_h)
            head_pose = {"angle-desc": desc, "angles": angles}

        x1, y1, x2, y2 = f.bbox.astype(int)

        # Add small margin to include entire face
        h, w = frame.shape[:2]
        margin = 20
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)

        face_img = frame[y1:y2, x1:x2]

        # Skip if face image is empty (invalid bbox)
        if face_img.size == 0:
            continue
        emotion = predict_emotion(face_img)


        entry = {
            "class": "face",
            "bbox": [x1, y1, x2, y2],
            "extra": {
                "recognized-name": name,
                "similarity": float(max_sim),
                "emotion": emotion,
                "position-desc": position_desc,
                "head-pose": head_pose
            }
        }
        detection_info.append(entry)

        if DISPLAY:
            cv2.putText(frame, f"{name} | {emotion}", (x1, max(10, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            for (x, y) in getattr(f, "kps", []):
                cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
            if head_pose is not None:
                cv2.putText(frame, head_pose["angle-desc"], (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)

    # --- Step 2: Draw non-person objects from YOLO
    for det in detections[:MAX_OBJECTS]:
        x1, y1, x2, y2, conf, cls_id = det
        cls_name = yolo.names[int(cls_id)]
        if cls_name not in ["person", "face"]:
            entry = {
                "class": cls_name,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": float(conf),
                "extra": {}
            }
            detection_info.append(entry)
            uz_name = COCO_UZBEK.get(cls_name, cls_name)

            if DISPLAY:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 1)
                cv2.putText(frame, uz_name, (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)


    return frame, detection_info


# ---------- RealSense helpers ----------
def _pick_color_profile(device, desired_fps=30):
    # Prefer 1280x720 @ desired_fps, else any BGR8 profile on the RGB sensor
    for sensor in device.sensors:
        try:
            if sensor.get_info(rs.camera_info.name) != "RGB Camera":
                continue
        except Exception:
            continue
        for p in sensor.profiles:
            v = p.as_video_stream_profile()
            if v and v.format() == rs.format.bgr8:
                if v.width() == 1280 and v.height() == 720 and v.fps() == desired_fps:
                    return v
    for sensor in device.sensors:
        try:
            if sensor.get_info(rs.camera_info.name) != "RGB Camera":
                continue
        except Exception:
            continue
        for p in sensor.profiles:
            v = p.as_video_stream_profile()
            if v and v.format() == rs.format.bgr8:
                return v
    return None

# ---------- RealSense capture thread ----------
def realsense_capture_loop(realsense=False):
    global _latest_frame, _latest_detections, _running, _pipeline
    pipeline = None
    cap = None
    try:
        if realsense:
            ctx = rs.context()
            if len(ctx.devices) == 0:
                print("[WARN] No RealSense device found; falling back to USB camera.")
                realsense = False
            else:
                dev = ctx.devices[0]
                try:
                    name = dev.get_info(rs.camera_info.name)
                except Exception:
                    name = "RealSense"
                print(f"[INFO] Using RealSense: {name}")
                pipeline = rs.pipeline()
                config = rs.config()

                prof = _pick_color_profile(dev, desired_fps=VIDEO_FPS)
                if prof is not None:
                    v = prof.as_video_stream_profile()
                    config.enable_stream(rs.stream.color, VIDEO_WIDTH, VIDEO_HEIGHT, rs.format.bgr8, v.fps())
                else:
                    # Let SDK auto-pick a valid color mode at desired FPS
                    config.enable_stream(rs.stream.color, 0, 0, rs.format.bgr8, VIDEO_FPS)

                try:
                    profile = pipeline.start(config)
                    _pipeline = pipeline
                    print("[INFO] RealSense pipeline started.")
                except Exception as e:
                    print(f"[WARN] RealSense start failed ({e}); falling back to USB camera.")
                    realsense = False
                    pipeline = None
                    _pipeline = None

        if not realsense:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, VIDEO_FPS)
            if not cap.isOpened():
                raise RuntimeError("USB camera could not be opened.")
            print("[INFO] USB camera pipeline started.")

        while _running:
            if pipeline is not None:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                color_image = np.asanyarray(color_frame.get_data())
            else:
                ret, color_image = cap.read()
                if not ret:
                    print("[WARN] USB camera read failed; retrying...")
                    time.sleep(0.01)
                    continue

            annotated, detections = detect_faces_and_objects(color_image)

            with _frame_lock:
                _latest_frame = annotated.copy()
                _latest_detections = detections

            if DISPLAY:
                # Optional overlay of face count
                cv2.putText(annotated, f"faces: {len(detections)}", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # Optional key handling only if HighGUI present
                try:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("[INFO] 'q' pressed - stopping.")
                        _running = False
                        break
                except Exception:
                    pass

    except Exception as e:
        print(f"[ERROR] RealSense capture loop exception: {e}")
    finally:
        try:
            if pipeline is not None:
                pipeline.stop()
                print("[INFO] RealSense pipeline stopped.")
        except Exception:
            pass
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        try:
            if DISPLAY:
                cv2.destroyAllWindows()
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
    with open(path, "r") as f:
        html = f.read()
    return web.Response(content_type="text/html", text=html)

async def offer(request):
    try:
        params = await request.json()
    except json.JSONDecodeError:
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

def to_serializable(obj):
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)  # fallback

async def detections_handler(request):
    with _frame_lock:
        data = _latest_detections.copy() if _latest_detections is not None else []
    return web.Response(
        content_type="application/json",
        text=json.dumps(data, default=to_serializable)
    )

async def on_shutdown(app):
    global _running, _pipeline
    print("[INFO] Shutting down server...")
    _running = False
    try:
        if _pipeline is not None:
            _pipeline.stop()
    except Exception:
        pass
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros, return_exceptions=True)
    pcs.clear()
    try:
        if DISPLAY:
            cv2.destroyAllWindows()
    except Exception:
        pass

# ---------- MJPEG video stream ----------
async def mjpeg_handler(request):
    boundary = "frame"
    response = web.StreamResponse(
        status=200, reason="OK",
        headers={"Content-Type": f"multipart/x-mixed-replace; boundary={boundary}"}
    )
    await response.prepare(request)

    try:
        while _running:
            with _frame_lock:
                frame = _latest_frame.copy() if _latest_frame is not None else None
            if frame is None:
                frame = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)

            ok, jpeg = cv2.imencode(".jpg", frame)
            if not ok:
                await asyncio.sleep(1.0 / VIDEO_FPS)
                continue
            data = jpeg.tobytes()

            try:
                await response.write(
                    b"--" + boundary.encode() + b"\r\n"
                    + b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(data)}\r\n\r\n".encode()
                    + data
                    + b"\r\n"
                )
            except (ConnectionResetError, RuntimeError):  # client disconnected / transport closing
                break

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


