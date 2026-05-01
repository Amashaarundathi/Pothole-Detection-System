from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import asyncio
import base64
import cv2
import logging
import numpy as np
import os
import re
import socket
import subprocess
import threading
import time
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Auto-free port before starting
# ─────────────────────────────────────────────
def free_port(port: int):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return
        result = subprocess.check_output(
            f'netstat -ano | findstr :{port} | findstr LISTENING',
            shell=True, text=True
        )
        for line in result.strip().splitlines():
            pid = line.strip().split()[-1]
            subprocess.call(f"taskkill /PID {pid} /F", shell=True)
            logger.info(f"Freed port {port} by killing PID {pid}")
        time.sleep(0.5)
    except Exception:
        pass

# ─────────────────────────────────────────────
# Try importing YOLO
# ─────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except (ImportError, OSError):
    YOLO_AVAILABLE = False
    logger.warning("ultralytics not available. Running in MOCK mode.")

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Global Variables
# ─────────────────────────────────────────────
detection_running = False
current_source = None
model = None
latest_frame = None
latest_detections = []
latest_alert = None
detection_thread = None

# ─────────────────────────────────────────────
# Load YOLO Model
# ─────────────────────────────────────────────
def load_model():
    global model
    if YOLO_AVAILABLE:
        try:
            model_path = "best.pt" if os.path.exists("best.pt") else "yolov8n.pt"
            model = YOLO(model_path)
            logger.info(f"Model loaded: {model_path}")
        except Exception as e:
            logger.error(f"Model load error: {e}")
            model = None
    else:
        model = None

# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

app = FastAPI(title="Pothole Detection API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Secure filename helper
# ─────────────────────────────────────────────
def secure_filename(filename: str) -> str:
    filename = os.path.basename(filename)
    filename = re.sub(r"[^\w.\-]", "_", filename)
    return filename

def safe_upload_path(filename: str) -> str | None:
    safe = secure_filename(filename)
    ext = os.path.splitext(safe)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return None
    full_path = os.path.realpath(os.path.join(UPLOAD_DIR, safe))
    upload_root = os.path.realpath(UPLOAD_DIR)
    if not full_path.startswith(upload_root + os.sep):
        return None
    return full_path

# ─────────────────────────────────────────────
# Image Preprocessing
# ─────────────────────────────────────────────
def preprocess_frame(frame):
    frame = cv2.resize(frame, (640, 640))
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    frame = cv2.filter2D(frame, -1, kernel)
    return frame

# ─────────────────────────────────────────────
# Mock Detection
# ─────────────────────────────────────────────
def mock_detections(frame_idx):
    detections = []
    if frame_idx % 30 < 15:
        detections = [
            {"id": 1, "location": "Center-Left", "severity": "High",
             "distance": round(2.3 + (frame_idx % 10) * 0.1, 1),
             "confidence": 0.87, "bbox": [120, 200, 280, 320]}
        ]
        if frame_idx % 60 < 20:
            detections.append({
                "id": 2, "location": "Right", "severity": "Medium",
                "distance": 4.1, "confidence": 0.72,
                "bbox": [400, 180, 520, 280]
            })
    return detections

# ─────────────────────────────────────────────
# Severity color map (BGR)
# ─────────────────────────────────────────────
SEVERITY_COLORS = {
    "High":   (0, 0, 255),
    "Medium": (0, 165, 255),
    "Low":    (0, 255, 0),
}

# ─────────────────────────────────────────────
# Draw line from bottom to bounding box
# ─────────────────────────────────────────────
def draw_distance_line(frame, bbox, distance):
    x1, y1, x2, y2 = bbox
    mid_x = (x1 + x2) // 2
    bottom_y = frame.shape[0]
    cv2.line(frame, (mid_x, bottom_y), (mid_x, y2), (255, 0, 0), 2)
    cv2.putText(frame, f"{distance}m", (mid_x + 5, bottom_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# ─────────────────────────────────────────────
# Detection Worker
# ─────────────────────────────────────────────
def run_detection(source):
    global detection_running, latest_frame, latest_detections, latest_alert  # ← fixed

    cap = cv2.VideoCapture(0 if source == "camera" else source)
    frame_idx = 0

    while detection_running:
        ret, frame = cap.read()
        if not ret:
            if source != "camera":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            time.sleep(0.03)
            continue

        frame = preprocess_frame(frame)
        detections = []

        if model is not None and YOLO_AVAILABLE:
            results = model(frame, imgsz=640, conf=0.4, verbose=False)
            for r in results:
                for i, box in enumerate(r.boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = model.names.get(cls, "pothole")
                    box_area = (x2 - x1) * (y2 - y1)
                    distance = round(500 / max((x2 - x1), 1), 1)
                    severity = "High" if box_area > 10000 else "Medium" if box_area > 4000 else "Low"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), SEVERITY_COLORS[severity], 2)
                    draw_distance_line(frame, [x1, y1, x2, y2], distance)
                    detections.append({
                        "id": i + 1, "location": label,
                        "severity": severity, "distance": distance,
                        "confidence": round(conf, 2),
                        "bbox": [x1, y1, x2, y2]
                    })
        else:
            detections = mock_detections(frame_idx)
            for d in detections:
                x1, y1, x2, y2 = d["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), SEVERITY_COLORS[d["severity"]], 2)
                draw_distance_line(frame, [x1, y1, x2, y2], d["distance"])

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        latest_frame = base64.b64encode(buf).decode("utf-8")
        latest_detections = detections

        # Determine alert level — latest_alert is now properly global
        severities = [d.get("severity") for d in detections]
        if "High" in severities:
            latest_alert = "High"
        elif "Medium" in severities:
            latest_alert = "Medium"
        elif "Low" in severities:
            latest_alert = "Low"
        else:
            latest_alert = None

        frame_idx += 1
        time.sleep(0.03)

    cap.release()

# ─────────────────────────────────────────────
# FastAPI Routes
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "model": "loaded" if model else "mock"}

@app.get("/api/status")
def get_status():
    return {"running": detection_running, "source": current_source,
            "model_loaded": model is not None, "yolo_available": YOLO_AVAILABLE}

@app.get("/api/analytics")
def get_analytics(period: str = "24h"):
    return {
        "period": period,
        "total_detected": len(latest_detections),
        "high_severity": sum(1 for d in latest_detections if d.get("severity") == "High"),
        "medium_severity": sum(1 for d in latest_detections if d.get("severity") == "Medium"),
        "low_severity": sum(1 for d in latest_detections if d.get("severity") == "Low"),
    }

@app.get("/api/locations")
def get_locations():
    return {"locations": [
        {"id": i + 1, "label": d.get("location", "Unknown"),
         "severity": d.get("severity"), "confidence": d.get("confidence"),
         "distance": d.get("distance")}
        for i, d in enumerate(latest_detections)
    ]}

@app.get("/api/stats")
def get_stats():
    return {
        "total_detected": len(latest_detections),
        "high_severity": sum(1 for d in latest_detections if d.get("severity") == "High"),
        "medium_severity": sum(1 for d in latest_detections if d.get("severity") == "Medium"),
        "low_severity": sum(1 for d in latest_detections if d.get("severity") == "Low"),
    }

@app.post("/api/detection/start")
def start_detection(source: str = "camera"):
    global detection_running, current_source, detection_thread
    if detection_running:
        return {"status": "already_running"}
    detection_running = True
    current_source = source
    detection_thread = threading.Thread(target=run_detection, args=(source,), daemon=True)
    detection_thread.start()
    return {"status": "started", "source": source}

@app.post("/api/detection/stop")
def stop_detection():
    global detection_running, current_source
    detection_running = False
    current_source = None
    return {"status": "stopped"}

@app.post("/api/upload-video")
async def upload_video(file: UploadFile = File(...)):
    path = safe_upload_path(file.filename)
    if path is None:
        return JSONResponse(status_code=400, content={"error": "Invalid filename or file type. Allowed: mp4, avi, mov, mkv, webm"})
    with open(path, "wb") as f:
        f.write(await file.read())
    return {"filename": os.path.basename(path), "path": path}

@app.post("/api/detection/start-video")
async def start_video(filename: str = Form(...)):
    global detection_running, current_source, detection_thread
    path = safe_upload_path(filename)
    if path is None:
        return JSONResponse(status_code=400, content={"error": "Invalid filename or file type"})
    if not os.path.exists(path):
        return JSONResponse(status_code=404, content={"error": "File not found"})
    if detection_running:
        detection_running = False
        time.sleep(0.5)
    detection_running = True
    current_source = path
    detection_thread = threading.Thread(target=run_detection, args=(path,), daemon=True)
    detection_thread.start()
    return {"status": "started", "source": path}

# ─────────────────────────────────────────────
# WebSocket Stream
# ─────────────────────────────────────────────
@app.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            if latest_frame:
                await ws.send_json({
                    "data": latest_frame,
                    "detections": latest_detections,
                    "alert": latest_alert,
                    "timestamp": time.time()
                })
            await asyncio.sleep(0.033)
    except WebSocketDisconnect:
        pass

# ─────────────────────────────────────────────
# Run Server
# ─────────────────────────────────────────────
if __name__ == "__main__":
    free_port(8000)
    uvicorn.run(app, host="0.0.0.0", port=8000)
