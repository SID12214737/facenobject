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
VIDEO_FPS = 29

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
                cv2.putText(frame, "H", (10, 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255))
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
