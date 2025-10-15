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

def udpclient():
    DEST_IP = "195.158.8.218"
    DEST_PORT = 9999
    MAX_DGRAM = 2**16 - 64
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cap = cv2.VideoCapture(0)
    print(f"[INFO] Sending to {DEST_IP}:{DEST_PORT}")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        _, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        data = pickle.dumps(encoded)

        for i in range(0, len(data), MAX_DGRAM):
            sock.sendto(data[i:i+MAX_DGRAM], (DEST_IP, DEST_PORT))
        sock.sendto(b'FRAME_END', (DEST_IP, DEST_PORT))

        time.sleep(0.03)

if __name__ == "__main__":
    udpclient()
