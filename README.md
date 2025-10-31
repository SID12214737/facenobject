# Robot Vision System

A real-time computer vision system with face recognition, emotion detection, head pose estimation, and object detection. Built for robotics applications with support for Intel RealSense cameras and standard USB cameras.

## Features

### Face Analysis
- **Face Recognition**: Database-backed face identification using InsightFace embeddings
- **Emotion Detection**: Real-time emotion classification (8 emotions in Uzbek)
- **Head Pose Estimation**: 3D head orientation tracking (yaw, pitch, roll)
- **Face Position Tracking**: Spatial location within frame

### Object Detection
- **YOLO Integration**: Real-time object detection (80+ classes)
- **Uzbek Translations**: Object labels in Uzbek language
- **Confidence Scoring**: Detection quality metrics

### Camera Support
- **Intel RealSense**: Native support for RealSense D400 series cameras
- **USB Cameras**: Fallback to standard webcams
- **Auto-detection**: Automatic camera selection

### Streaming & API
- **WebRTC Streaming**: Low-latency video streaming
- **MJPEG Support**: Compatible fallback streaming format
- **REST API**: Full-featured HTTP API for face management
- **Real-time Detections**: JSON endpoint for detection metadata

## Installation

### Prerequisites

```bash
# System dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev
sudo apt-get install -y libopencv-dev python3-opencv
sudo apt-get install -y librealsense2-dev librealsense2-utils

# For RealSense camera (optional)
# Follow: https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md
```

### Python Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
opencv-python>=4.8.0
numpy>=1.24.0
pyrealsense2>=2.54.0
insightface>=0.7.3
ultralytics>=8.0.0
onnxruntime-gpu>=1.16.0  # or onnxruntime for CPU
aiohttp>=3.9.0
aiortc>=1.6.0
aiohttp-cors>=0.7.0
```

### Model Files

1. **YOLOv8 Model** (auto-downloaded on first run):
   ```bash
   # Downloads automatically via ultralytics
   ```

2. **Emotion Detection Model**:
   ```bash
   wget https://huggingface.co/webml/models-moved/resolve/0e73dc31942fbdbbd135d85be5e5321eee88a826/emotion-ferplus-8.onnx -O emotion-ferplus-8.onnx
   ```

3. **InsightFace Models** (auto-downloaded on first run):
   ```bash
   # Downloads automatically via insightface
   ```

## Usage

### Starting the Server

**Full version with object detection:**
```bash
python3 robot_vision.py
```

**Face-only version (no YOLO):**
```bash
python3 robot_vision_noYolo.py
```

The server will start on `http://0.0.0.0:8088`

### Command Line Interface

The system includes an interactive CLI for face management:

```
FACE MANAGEMENT COMMANDS:
  name <id> <name>       - Name an unknown person by their ID
                          Example: name 1 John
  rename <id> <name>     - Rename existing person in database
                          Example: rename 5 Jane
  list                   - List all known faces in database
  unknown                - Show current unknown faces
  delete <id>            - Delete face from database
  reload                 - Reload database
  quit                   - Exit program
```

### API Endpoints

#### Video Streaming

**WebRTC Stream:**
```javascript
POST /offer
Content-Type: application/json

{
  "sdp": "<SDP_OFFER>",
  "type": "offer"
}
```

**MJPEG Stream:**
```
GET /video
```

#### Face Management

**Register New Face:**
```bash
POST /ragister-name
Content-Type: application/json

{
  "unknown_id": 1,
  "name": "John Doe"
}
```

**Remove Face:**
```bash
POST /remove-name
Content-Type: application/json

{
  "face_id": 5
}
```

**List All Faces:**
```bash
GET /list-names

Response:
{
  "success": true,
  "known_faces": [
    {
      "id": 1,
      "name": "John Doe",
      "created_at": "2024-10-31 10:30:00",
      "last_seen": "2024-10-31 12:45:00"
    }
  ],
  "unknown_faces": [
    {"unknown_id": 2}
  ],
  "total_known": 1,
  "total_unknown": 1
}
```

**Clear Unknown Faces:**
```bash
POST /clear-unknown

Response:
{
  "success": true,
  "cleared_count": 3,
  "message": "Successfully cleared 3 unknown face(s)"
}
```

#### Detection Data

**Get Current Detections:**
```bash
GET /detections

Response:
[
  {
    "class": "face",
    "bbox": [100, 150, 250, 300],
    "extra": {
      "recognized-name": "John Doe",
      "face-id": 1,
      "similarity": 0.87,
      "emotion": "xursand",
      "position-desc": "center-center",
      "head-pose": {
        "angle-desc": "Tog'riga & Tog'riga",
        "angles": {
          "yaw": 5.2,
          "pitch": -3.1,
          "roll": 1.8
        }
      }
    }
  },
  {
    "class": "laptop",
    "bbox": [300, 200, 500, 400],
    "confidence": 0.92,
    "extra": {}
  }
]
```

## Configuration

Edit these constants in the script:

```python
MAX_FACES = 5              # Maximum faces to detect per frame
MAX_OBJECTS = 5            # Maximum objects to detect per frame
VIDEO_WIDTH = 640          # Video width
VIDEO_HEIGHT = 480         # Video height
VIDEO_FPS = 30             # Target FPS
HEADLESS = False           # Set True for server without display
```

## Database

Face data is stored in `faces.db` (SQLite):

**Schema:**
```sql
CREATE TABLE faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    encoding BLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Backup:**
```bash
cp faces.db faces_backup.db
```

**Reset:**
```bash
rm faces.db
# Database will be recreated on next run
```

## Emotion Labels (Uzbek)

| English | Uzbek |
|---------|-------|
| neutral | neytral |
| happiness | xursand |
| surprise | hayrat |
| sadness | xafa |
| anger | g'azab |
| disgust | jirkanch |
| fear | qo'rquv |
| contempt | mensimaslik |

## Object Classes (Uzbek)

Common objects with Uzbek translations (80+ total):
- odam (person)
- mashina (car)
- telefon (cell phone)
- noutbuk (laptop)
- kitob (book)
- And many more...

## Performance Optimization

### GPU Acceleration
```python
# Edit PROVIDERS list for your hardware:
PROVIDERS = [
    "TensorrtExecutionProvider",  # NVIDIA TensorRT
    "CUDAExecutionProvider",      # NVIDIA CUDA
    "CPUExecutionProvider"        # CPU fallback
]
```

### CPU-Only Mode
```bash
pip install onnxruntime  # instead of onnxruntime-gpu
```

### Reduce Frame Rate
```python
VIDEO_FPS = 15  # Lower FPS for slower systems
```

## Troubleshooting

### RealSense Not Detected
```bash
# Check camera connection
rs-enumerate-devices

# Test camera
realsense-viewer
```

### CUDA/GPU Issues
```bash
# Verify CUDA installation
nvidia-smi

# Install CUDA toolkit
# https://developer.nvidia.com/cuda-downloads
```

### Port Already in Use
```python
# Change port in script:
web.run_app(webapp, host="0.0.0.0", port=8089)  # Change 8088 to 8089
```

### Low FPS
1. Reduce `MAX_FACES` and `MAX_OBJECTS`
2. Lower `VIDEO_WIDTH` and `VIDEO_HEIGHT`
3. Disable object detection (use `robot_vision_noYolo.py`)
4. Enable GPU acceleration

## Web Client Example

Create `index.html` for browser access:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Robot Vision</title>
</head>
<body>
    <h1>Robot Vision Stream</h1>
    <img src="/video" width="640" height="480" />
    
    <script>
        // Fetch detections every 500ms
        setInterval(async () => {
            const response = await fetch('/detections');
            const data = await response.json();
            console.log('Detections:', data);
        }, 500);
    </script>
</body>
</html>
```

## Architecture

```
┌─────────────────────────────────────────┐
│         Camera Input Thread             │
│  (RealSense / USB Camera Capture)       │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│      Detection Pipeline                 │
│  - InsightFace (face detection)         │
│  - YOLO (object detection)              │
│  - Emotion Detection (ONNX)             │
│  - Head Pose Estimation                 │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│      Face Recognition                   │
│  - Embedding comparison                 │
│  - SQLite database                      │
│  - Unknown face tracking                │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│      Annotation & Storage               │
│  - Draw bounding boxes                  │
│  - Add labels                           │
│  - Update shared frame                  │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│      Streaming Servers                  │
│  - WebRTC (low latency)                 │
│  - MJPEG (compatibility)                │
│  - REST API                             │
└─────────────────────────────────────────┘
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Credits

- **InsightFace**: Face recognition models
- **Ultralytics YOLOv8**: Object detection
- **Microsoft Emotion FERPlus**: Emotion detection
- **Intel RealSense**: Camera SDK
- **aiortc**: WebRTC implementation
