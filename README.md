# 🚗💤 DrowSim

> **Real-time driver drowsiness detection** using a privacy-preserving edge-cloud architecture — Raspberry Pi 4 client + BiLSTM-Attention inference server on AWS EC2.

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-0097A7?style=flat-square&logo=google&logoColor=white)](https://mediapipe.dev)
[![AWS](https://img.shields.io/badge/AWS-EC2-FF9900?style=flat-square&logo=amazonaws&logoColor=white)](https://aws.amazon.com)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Model Architecture](#-model-architecture)
- [Performance Results](#-performance-results)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [API Reference](#-api-reference)
- [Cloud Deployment](#-cloud-deployment)
- [How It Works](#-how-it-works)
- [Dataset](#-dataset)
- [Acknowledgements](#-acknowledgements)

---

## 🌟 Overview

DrowSim is a final-year project (FYP) system that detects driver drowsiness in real time using a **hybrid detection pipeline** combining deterministic rule-based logic with a deep learning **BiLSTM + Attention** classifier.

The system runs the computer vision pipeline on a **Raspberry Pi 4** (edge device), transmits only 4 numerical features per frame to a **FastAPI server on AWS EC2**, and returns a drowsiness state in under **500 ms** end-to-end — without ever sending video or images to the cloud.

**Detected states:**
| State | Detection Method | Alert |
|---|---|---|
| 🟢 Alert | BiLSTM model | — |
| 🔴 Microsleep | Rule-based (EAR threshold) | 1500 Hz alarm |
| 🟡 Yawning | BiLSTM model | 900 Hz warning |

---

## ✨ Key Features

- **Privacy-preserving** — only 4 geometric features transmitted per frame, zero raw images or video sent to the cloud
- **Hybrid detection** — deterministic rule-based microsleep detector (safety-critical, interpretable) + learned BiLSTM classifier for alert/yawning
- **Per-user calibration** — EAR baseline computed per session from the first 2 seconds to account for inter-driver variability
- **TrueEAR** — pose-invariant eye aspect ratio computed from 6 landmarks, robust to head rotation
- **Temporal smoothing** — 9-frame majority-vote buffer eliminates single-frame prediction oscillation
- **Hysteresis counter** — `+1 if closed, −2 if open` logic prevents blink-triggered false alarms
- **Production deployment** — HTTPS via Let's Encrypt, Nginx reverse proxy, Elastic IP, custom domain `drowsim.fyp.systems`
- **Graceful degradation** — client continues operating safely when server is unreachable

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Raspberry Pi 4 (Client)                    │
│                                                                 │
│  Camera → OpenCV (160×120) → MediaPipe Face Mesh               │
│       ↓                                                         │
│  5 Landmarks [33, 263, 1, 78, 308]                              │
│       ↓                                                         │
│  TrueEAR + Feature Extraction → [EAR, MAR, nose_angle,         │
│                                   eye_angle]  (4 values)        │
│       ↓                                                         │
│  HTTPS POST → https://drowsim.fyp.systems/infer                 │
│       ↓                                                         │
│  State + Confidence ← JSON Response                             │
│       ↓                                                         │
│  Audio Alert (pygame) + Visual Overlay (OpenCV)                 │
└─────────────────────────────────────────────────────────────────┘
                              │  TLS/HTTPS
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AWS EC2 (Cloud Server)                        │
│                                                                 │
│  Nginx (reverse proxy + SSL termination)                        │
│       ↓                                                         │
│  FastAPI /infer endpoint                                        │
│       ↓                                                         │
│  Phase 1: EAR Calibration (first ~2 s)                          │
│  Phase 2: Rule-based Microsleep (EAR < 0.75 × baseline)        │
│  Phase 3: BiLSTM-Attention Inference (alert / yawning)          │
│       ↓                                                         │
│  {"state": "alert|microsleep|yawning", "confidence": [...]}     │
└─────────────────────────────────────────────────────────────────┘
```

**Data transmitted per frame:** `~200 bytes` (JSON with 5 landmark coordinates + EAR value)
**No video, no images, no biometric data stored.**

---

## 🧠 Model Architecture

```
Input: (batch, 32 frames, 4 features)
           ↓
    BiLSTM (hidden=64, layers=2, bidirectional)
      → output: (batch, 32, 128)
           ↓
    Attention Layer  [Linear(128→1) + Softmax]
      → weighted context vector: (batch, 128)
           ↓
    Dropout (p=0.3)
           ↓
    Fully Connected  128 → 3
           ↓
    Softmax
           ↓
    Output: [P(alert), P(microsleep), P(yawning)]
```

| Parameter | Value |
|---|---|
| Input features | 4 (EAR, MAR, nose\_angle, eye\_angle) |
| Sequence length | 32 frames |
| Hidden size | 64 |
| LSTM layers | 2 |
| Directions | Bidirectional |
| Attention | Single-head (learned weights) |
| Dropout | 0.3 |
| Output classes | 3 (alert, microsleep, yawning) |
| Loss function | Focal Loss (γ=2) |
| Optimizer | Adam |
| Parameters | ~100K |

**Design rationale:** The minimal 4-feature input was intentional — it reduces data transmission by ~95% compared to sending full landmark vectors while maintaining competitive accuracy. Microsleep is intentionally excluded from the model's decision scope; it is always handled by the deterministic rule.

---

## 📊 Performance Results

### Model Validation (Person-Disjoint Test Set)

Evaluated on **1,094 sequences** from **7 unseen drivers** (not seen during training):

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Alert | 0.9645 | 0.9645 | 0.9645 | 844 |
| Microsleep | 0.8466 | 0.8466 | 0.8466 | 163 |
| Yawning | 0.9425 | 0.9425 | 0.9425 | 87 |
| **Macro avg** | **0.9179** | **0.9179** | **0.9179** | 1094 |
| **Weighted avg** | **0.9452** | **0.9452** | **0.9452** | 1094 |

**Overall accuracy: 94.52%** ✅ (requirement: ≥ 85%)

> The lower Microsleep F1 (0.847) is by design — microsleep is the safety-critical class handled by the deterministic rule-based detector independently of the model.

### Robustness Testing (Gaussian Noise, σ = 0.02)

| Condition | Accuracy |
|---|---|
| Clean features | 0.9452 |
| With noise (σ=0.02) | 0.9461 |

Accuracy drift < 0.1% — model is stable under sensor perturbation and does not overfit to exact numerical values.

### Latency & Performance

| Metric | Measured | Requirement |
|---|---|---|
| BiLSTM inference (CPU) | **1.77 ± 0.15 ms** | < 500 ms |
| Max inference latency | 2.73 ms | < 500 ms |
| RPi landmark extraction @ 10 FPS | ~30 ms/frame | < 100 ms |
| HTTPS round-trip (AWS) | 200–400 ms | < 5 s timeout |
| End-to-end alert latency | **< 500 ms** | < 500 ms ✅ |

### End-to-End Integration Tests

| Test | Scenario | Result |
|---|---|---|
| E2E-01 | Face detection → state display | ✅ PASS |
| E2E-02 | HTTPS transmission (Wireshark verified) | ✅ PASS |
| E2E-03 | Microsleep → RULE path | ✅ PASS |
| E2E-04 | Yawning → MODEL path (confidence > 0.5) | ✅ PASS |
| E2E-05 | Audio alert playback (1500 Hz / 900 Hz) | ✅ PASS |
| E2E-06 | `/health` endpoint returns 200 OK | ✅ PASS |
| E2E-07 | HTTPS certificate (Let's Encrypt, verified via openssl) | ✅ PASS |
| E2E-08 | Privacy: no raw data on disk after shutdown | ✅ PASS |

---

## 📁 Project Structure

```
DrowSim-FYP/
│
├── server/                         # FastAPI inference server (AWS EC2)
│   ├── server4.py                  # Main server — BiLSTM + /infer endpoint
│   └── requirements_server.txt
│
├── client/                         # Raspberry Pi 4 client
│   ├── client.py                   # Main client — capture, extract, transmit, alert
│   └── requirements_client.txt
│
├── module/                       
│   ├── BiLSTM4.pth             
│   ├── BiLSTM4FetCluResult                 
│
├── docs/
│   └── architecture_diagram.png
│
├── .gitignore
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.9+
- Raspberry Pi 4 (4GB RAM recommended) with camera module — for client
- Ubuntu 22.04 server (AWS EC2 t2.micro or better) — for server

---

### 🖥️ Server Setup (AWS EC2 / Any Linux Server)

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/DrowSim-FYP.git
cd DrowSim-FYP/server

# 2. Install dependencies
pip install -r requirements_server.txt

# 3. Ensure model weights are present
ls BiLSTM4.pth   # should exist

# 4. Run the server
uvicorn server4:app --host 0.0.0.0 --port 8000

# 5. (Optional) Run with Nginx + HTTPS — see Cloud Deployment section
```

**Server dependencies:**
```
fastapi
uvicorn[standard]
torch
numpy
pydantic
```

---

### 🍓 Client Setup (Raspberry Pi 4)

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/DrowSim-FYP.git
cd DrowSim-FYP/client

# 2. Install dependencies
pip install -r requirements_client.txt

# 3. Configure server URL in client.py
# SERVER_URL = "https://drowsim.fyp.systems/infer"   # production
# SERVER_URL = "http://<your-ip>:8000/infer"          # local testing

# 4. Run the client
python client.py
```

**Client dependencies:**
```
opencv-python
mediapipe
requests
pygame
numpy
```

---

### 🧪 Training (Optional — reproduce the model)

```bash
cd DrowSim-FYP/training

# Install training dependencies
pip install torch numpy scikit-learn pandas matplotlib seaborn

# Prepare FL3D dataset (see Dataset section)

# Train
python train.py
```

---

## 📡 API Reference

### Base URL
```
https://drowsim.fyp.systems
```

---

### `POST /infer`

Main inference endpoint. Accepts per-frame data and returns drowsiness state.

**Request body:**
```json
{
  "landmarks_5": [
    [x1, y1, z1],
    [x2, y2, z2],
    [x3, y3, z3],
    [x4, y4, z4],
    [x5, y5, z5]
  ],
  "true_ear": 0.312,
  "fps": 10.0
}
```

| Field | Type | Description |
|---|---|---|
| `landmarks_5` | list[list[float]] | 5 facial landmarks: [left\_eye, right\_eye, nose, mouth\_L, mouth\_R] — MediaPipe indices [33, 263, 1, 78, 308] |
| `true_ear` | float | Pre-computed TrueEAR value from the client |
| `fps` | float (optional) | Client FPS — used to scale calibration and microsleep thresholds. Default: 30.0 |

**Response:**
```json
{
  "state": "alert",
  "confidence": [0.9512, 0.0231, 0.0257],
  "source": "MODEL",
  "debug": {
    "true_ear": 0.312,
    "ear_th": 0.2415,
    "model_raw": "alert"
  }
}
```

| Field | Values | Description |
|---|---|---|
| `state` | `"calibrating"` / `"buffering"` / `"alert"` / `"microsleep"` / `"yawning"` | Current driver state |
| `confidence` | `[float, float, float]` | Softmax probabilities for [alert, microsleep, yawning] |
| `source` | `"CALIB"` / `"BUFFER"` / `"RULE"` / `"MODEL"` | Which component produced this decision |

**State lifecycle:**

```
Request 1–N  →  "calibrating"  (EAR baseline being computed)
Request N–M  →  "buffering"    (filling 32-frame sequence window)
Request M+   →  "alert" / "microsleep" / "yawning"
```

---

### `GET /health`

Health check endpoint.

```bash
curl https://drowsim.fyp.systems/health
```

```json
{
  "status": "ok",
  "device": "cpu",
  "model": "BiLSTM4.pth"
}
```

---

## ☁️ Cloud Deployment

The production server runs on **AWS EC2 (Ubuntu 22.04)** with the following stack:

```
Internet → Route 53 (DNS) → Elastic IP
                                ↓
                          AWS EC2 Instance
                                ↓
                     Nginx (port 443, SSL termination)
                       ↑ Let's Encrypt certificate
                                ↓
                    uvicorn (localhost:8000)
                                ↓
                         FastAPI /infer
```

### Nginx Configuration (`/etc/nginx/sites-available/drowsim`)

```nginx
server {
    listen 443 ssl;
    server_name drowsim.fyp.systems;

    ssl_certificate     /etc/letsencrypt/live/drowsim.fyp.systems/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/drowsim.fyp.systems/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}

server {
    listen 80;
    server_name drowsim.fyp.systems;
    return 301 https://$host$request_uri;
}
```

### SSL Certificate (Let's Encrypt)

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d drowsim.fyp.systems
```

### Run Server as a systemd Service

```bash
# /etc/systemd/system/drowsim.service
[Unit]
Description=DrowSim FastAPI Server
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/DrowSim-FYP/server
ExecStart=/usr/bin/python3 -m uvicorn server4:app --host 127.0.0.1 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable drowsim
sudo systemctl start drowsim
sudo systemctl status drowsim
```

---

## 🔬 How It Works

### Phase 1 — EAR Calibration
During the first ~2 seconds, the server collects TrueEAR values and computes a **personal baseline** using the median. The detection threshold is set to `EAR_TH = 0.75 × EAR_BASE`. This adapts to each driver's natural eye openness without requiring manual configuration.

### Phase 2 — Rule-Based Microsleep Detection
Every frame, if `TrueEAR < EAR_TH`, a counter increments by 1; otherwise it decrements by 2 (hysteresis to ignore blinks). If the counter reaches `~0.8 × fps` frames (~24 frames at 30 FPS), the server immediately returns `"microsleep"` with `source="RULE"`. The model is bypassed entirely for this state.

### Phase 3 — BiLSTM Temporal Classification
Once 32 frames have accumulated in the sliding window, the sequence is passed through the BiLSTM-Attention model. The attention mechanism weights each frame's contribution based on learned temporal importance. The raw prediction is smoothed using a 9-frame majority-vote buffer. The model cannot override a microsleep verdict from Phase 2.

### TrueEAR Computation
Instead of the standard 6-point EAR formula, DrowSim uses a simplified 3-point formula from MediaPipe's sparse landmark set that is **invariant to lateral head rotation**, computed as:

```
EAR = dist(left_eye, nose) / (dist(left_eye, right_eye) + ε)
```

---

## 📂 Dataset

The model was trained on the **FL3D (Facial Landmarks for Drowsiness Detection)** dataset, which contains annotated facial landmark sequences from multiple drivers under night-time driving conditions.

- **Training:** person-disjoint split — unseen drivers reserved for testing
- **Class distribution:** Alert (majority), Microsleep, Yawning
- **Imbalance handling:** Focal Loss (γ=2) during training

> Dataset is not included in this repository. Please refer to the original FL3D dataset source.

---

## 🎓 Academic Context

This project was developed as a Final Year Project (FYP) for the degree of **Bachelor of Information Technology** at **Universiti Kebangsaan Malaysia (UKM)**, Faculty of Information Science and Technology (FTSM), under the course code **TK/TM/TU/TH4086**.

**Live endpoint:** https://drowsim.fyp.systems/health

---

## 🙏 Acknowledgements

- [MediaPipe](https://mediapipe.dev) — facial landmark detection
- [FL3D Dataset](https://github.com/) — drowsiness detection benchmark
- [FastAPI](https://fastapi.tiangolo.com) — async API framework
- [PyTorch](https://pytorch.org) — deep learning framework
- [Let's Encrypt](https://letsencrypt.org) — free SSL certificates


---

<div align="center">
  <sub>Built with ❤️ as a Final Year Project @ UKM FTSM</sub>
</div>
