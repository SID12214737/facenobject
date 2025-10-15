import cv2
from video_source import get_video_source
from main import detect_faces_and_objects

try:
    source = get_video_source(mode="socket")  # or "socket", "ros2", or "camera"
except Exception as e:
    print(f"[ERROR] Could not initialize video source: {e}")
    exit(1)

while True:
    print("[DEBUG] Reading from source.")
    frame = source.read()

    print("[DEBUG] Frame read.")
    if frame is None:
        break

    print("[DEBUG] Frame received.")
    # Pass to your detection pipeline
    processed = detect_faces_and_objects(frame)

    cv2.imshow("Stream", processed)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

source.release()
cv2.destroyAllWindows()

print("[INFO] Application stopped.")