# 🔍 PotholeAI — Road Intelligence System

YOLOv8 + FastAPI + React — Real-time pothole detection system.

## 📁 Structure

```
pothole-detection/
├── backend/
│   ├── app.py              # FastAPI + WebSocket stream + detection engine
│   ├── requirements.txt
│   └── best.pt             # (place your trained YOLOv8 model here)
└── frontend/
    ├── index.html
    ├── package.json
    ├── vite.config.js
    ├── tailwind.config.js
    ├── postcss.config.js
    └── src/
        ├── main.jsx
        ├── App.jsx
        ├── index.css
        └── components/
            └── PotholeDetectionFrontend.jsx
```

## 🚀 Quick Start

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
python app.py
# → http://localhost:8000
```

> If `best.pt` is not found, it falls back to `yolov8n.pt`. If ultralytics is not installed, it runs in **MOCK mode** for UI testing.

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

## 🎯 Features

- **Live Camera Detection** — Real-time webcam feed with YOLOv8 inference
- **Video File Detection** — Upload and process video files
- **WebSocket Streaming** — Low-latency frame delivery to browser
- **Severity Classification** — High / Medium / Low based on bounding box area
- **Detection Log** — Timestamped event history
- **System Status Panel** — Live backend connection & model status
- **Mock Mode** — UI fully testable without a GPU or model file

## 🔧 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/status` | Detection & model status |
| POST | `/api/detection/start?source=camera` | Start camera detection |
| POST | `/api/detection/stop` | Stop detection |
| POST | `/api/upload-video` | Upload video file |
| POST | `/api/detection/start-video` | Start video detection |
| WS | `/ws/stream` | WebSocket frame + detection stream |

## 📦 Requirements

- Python 3.9+
- Node.js 18+
- (Optional) CUDA-enabled GPU for fast inference
- (Optional) `best.pt` — your trained YOLOv8 pothole model
