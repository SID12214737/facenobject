class VideoSource:
    def read(self):
        """Return the next frame (BGR np.array) or None if no frame."""
        raise NotImplementedError

import cv2

class CameraSource(VideoSource):
    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)

    def read(self):
        ret, frame = self.cap.read()
        return frame if ret else None

import socket, pickle, struct, cv2, numpy as np

class SocketSource(VideoSource):
    def __init__(self, host='0.0.0.0', port=9999):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((host, port))
        self.server_socket.listen(1)
        print("[INFO] Waiting for connection...")
        self.conn, _ = self.server_socket.accept()
        print("[INFO] Client connected.")

    def read(self):
        data = b""
        payload_size = struct.calcsize("Q")
        while len(data) < payload_size:
            packet = self.conn.recv(4*1024)
            if not packet: return None
            data += packet

        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        while len(data) < msg_size:
            data += self.conn.recv(4*1024)

        frame_data = data[:msg_size]
        data = data[msg_size:]
        frame = pickle.loads(frame_data)
        return cv2.imdecode(frame, cv2.IMREAD_COLOR)

# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# import threading, queue

# class ROS2ImageSource(VideoSource, Node):
#     def __init__(self, topic='/camera/image_raw'):
#         rclpy.init(args=None)
#         Node.__init__(self, 'ros2_video_source')
#         self.bridge = CvBridge()
#         self.frame_queue = queue.Queue(maxsize=1)
#         self.create_subscription(Image, topic, self.callback, 10)
#         self.thread = threading.Thread(target=rclpy.spin, args=(self,), daemon=True)
#         self.thread.start()

#     def callback(self, msg):
#         frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
#         if self.frame_queue.full():
#             _ = self.frame_queue.get_nowait()
#         self.frame_queue.put(frame)

#     def read(self):
#         return self.frame_queue.get()
