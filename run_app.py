import cv2
from video_source import get_video_source
from main import detect_faces_and_objects

try:
    source = get_video_source(mode="udp-client")  # or "socket", "ros2", or "camera"
except Exception as e:
    print(f"[ERROR] Could not initialize video source: {e}")
    exit(1)

while True:
    frame = source.read()

    if frame is None:
        print("[WARN] No frame received, quitting...")
        break

    # Pass to your detection pipeline
    processed = detect_faces_and_objects(frame)

    cv2.imshow("Stream", processed)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Quitting...")
        break

source.release()
cv2.destroyAllWindows()

print("[INFO] Application stopped.")