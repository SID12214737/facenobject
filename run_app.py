from video_source import CameraSource, SocketSource #, ROS2ImageSource


def get_video_source(mode="camera"):
    if mode == "camera":
        return CameraSource(0)
    elif mode == "socket":
        return SocketSource('0.0.0.0', 9999)
    # elif mode == "ros2":
    #     return ROS2ImageSource('/camera/image_raw')
    else:
        raise ValueError("Invalid mode")

try:
    source = get_video_source(mode="socket")  # or "ros2", or "camera"
except Exception as e:
    print(f"[ERROR] Could not initialize video source: {e}")
    exit(1)


while True:
    frame = source.read()
    if frame is None:
        break

    # Pass to your detection pipeline
    processed = detect_faces_and_objects(frame)

    cv2.imshow("Stream", processed)
    if cv2.waitKey(1) == 27:
        break
