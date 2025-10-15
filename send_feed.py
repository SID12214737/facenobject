import cv2
import socket
import struct
import pickle
import time

SERVER_IP = '0.0.0.0'  # 🔹 change to the robot/server IP
SERVER_PORT = 9999

def start_client():
    while True:
        try:
            print(f"[INFO] Connecting to server {SERVER_IP}:{SERVER_PORT} ...")
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((SERVER_IP, SERVER_PORT))
            print("[INFO] Connected to server.")

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("[ERROR] Cannot access camera.")
                client_socket.close()
                time.sleep(3)
                continue

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[WARN] Frame capture failed.")
                    break

                # Encode frame as JPEG to reduce size
                _, encoded_frame = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

                # Serialize data
                data = pickle.dumps(encoded_frame)
                message = struct.pack(">L", len(data)) + data

                try:
                    client_socket.sendall(message)
                except (BrokenPipeError, ConnectionResetError):
                    print("[WARN] Lost connection to server.")
                    break

            cap.release()
            client_socket.close()
            print("[INFO] Reconnecting in 3 seconds...")
            time.sleep(3)

        except Exception as e:
            print(f"[ERROR] {e}")
            time.sleep(3)

if __name__ == "__main__":
    start_client()
