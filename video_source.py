import cv2
import socket, pickle, struct, cv2, numpy as np

class VideoSource:
    def read(self):
        """Return the next frame (BGR np.array) or None if no frame."""
        raise NotImplementedError
    def release(self):
        """Release any resources if needed."""
        pass

class CameraSource(VideoSource):
    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)

    def read(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        self.cap.release()

class SocketSource(VideoSource):
    def __init__(self, host='0.0.0.0', port=9998):
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
        print(f"[DEBUG] Expecting frame of size: {msg_size} bytes")
        while len(data) < msg_size:
            data += self.conn.recv(4*1024)
            print('stuck')
        print(f"[DEBUG] Received {len(data)} bytes of frame data")
        frame_data = data[:msg_size]
        data = data[msg_size:]
        frame = pickle.loads(frame_data)
        return cv2.imdecode(frame, cv2.IMREAD_COLOR)
    
    def release(self):
        self.conn.close()
        self.server_socket.close()

class SocketClientSource(VideoSource):
    def __init__(self, host='192.168.0.42', port=9998):  # robot’s IP
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print(f"[INFO] Connecting to {host}:{port}...")
        self.client_socket.connect((host, port))
        print("[INFO] Connected to video server.")

    def read(self):
        data = b""
        payload_size = struct.calcsize("Q")

        while len(data) < payload_size:
            packet = self.client_socket.recv(4096)
            if not packet:
                return None
            data += packet

        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        while len(data) < msg_size:
            data += self.client_socket.recv(4096)

        frame_data = data[:msg_size]
        frame = pickle.loads(frame_data)
        return cv2.imdecode(frame, cv2.IMREAD_COLOR)
    
    def release(self):
        self.client_socket.close()  


class UDPSocketSource(VideoSource):
    def __init__(self, host='0.0.0.0', port=9999, max_dgram=2**16 - 64):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.sock.settimeout(0.5)
        self.max_dgram = max_dgram
        self.buffer = b""
        print(f"[INFO] Listening for UDP stream on {host}:{port}")

    def read(self):
        try:
            while True:
                segment, _ = self.sock.recvfrom(self.max_dgram)
                if segment == b'FRAME_END':
                    if not self.buffer:
                        return None
                    frame = pickle.loads(self.buffer)
                    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                    self.buffer = b""
                    return frame
                else:
                    self.buffer += segment
        except socket.timeout:
            return None
        except Exception as e:
            print("[ERROR]", e)
            return None

    def release(self):
        self.sock.close()

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

def get_video_source(mode="camera"):
    if mode == "camera":
        return CameraSource(0)
    elif mode == "socket":
        return SocketSource('0.0.0.0', 9999)
    elif mode == "socket-client":
        return SocketClientSource('0.0.0.0', 9999)
    elif mode == "udp-client":
        return UDPSocketSource('0.0.0.0', 9999)
    # elif mode == "ros2":
    #     return ROS2ImageSource('/camera/image_raw')
    else:
        raise ValueError("Invalid mode")