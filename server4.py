"""
DrowSim — FastAPI Inference Server
Model  : BiLSTM + Attention  (input_size=4, hidden_size=64, num_classes=3)
Model file : BiLSTM4.pth

Run:
    uvicorn server:app --host 0.0.0.0 --port 8000
"""

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel
from collections import deque


# ──────────────────────────────────────────────
# Model — must match training architecture exactly
# input_size=4, hidden_size=64, num_layers=2, num_classes=3
# ──────────────────────────────────────────────
class BiLSTM_Attention(nn.Module):
    def __init__(self, input_size=4, hidden_size=64,
                 num_layers=2, num_classes=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        out, _  = self.lstm(x)
        weights = torch.softmax(self.attention(out), dim=1)
        context = (out * weights).sum(dim=1)
        return self.fc(self.dropout(context))


# ──────────────────────────────────────────────
# Feature computation — 4 features from 5 points
# Point order: [left_eye, right_eye, nose, mouth_L, mouth_R]
# MediaPipe indices used by client: [33, 263, 1, 78, 308]
# ──────────────────────────────────────────────
def euclidean(a, b):
    return float(np.linalg.norm(
        np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)
    ))

def compute_features_4(pts5):
    """
    Returns [EAR, MAR, nose_angle, eye_angle] from 5 landmark points.
    Returns [0.0, 0.0, 0.0, 0.0] if input is invalid.
    """
    if pts5 is None or len(pts5) != 5:
        return [0.0, 0.0, 0.0, 0.0]

    p1, p2, p3, p4, p5 = [np.array(p, dtype=np.float32) for p in pts5]

    EAR        = euclidean(p1, p3) / (euclidean(p1, p2) + 1e-6)
    mouth_h    = euclidean(p3, p4) + euclidean(p3, p5)
    MAR        = mouth_h / (euclidean(p4, p5) + 1e-6)
    nose_angle = float(np.arctan2(p3[1] - p1[1], p3[0] - p1[0]))
    eye_angle  = float(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))

    return [EAR, MAR, nose_angle, eye_angle]


# ──────────────────────────────────────────────
# Load model
# ──────────────────────────────────────────────
app    = FastAPI(title="DrowSim Inference Server")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "BiLSTM4.pth"

model = BiLSTM_Attention(input_size=4, hidden_size=64,
                         num_layers=2, num_classes=3).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print(f"Model loaded  : {MODEL_PATH}")
print(f"Device        : {device}")


# ──────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────
CLASS_NAMES    = ["alert", "microsleep", "yawning"]
SEQ_LEN        = 32

features_buffer = deque(maxlen=SEQ_LEN)
pred_buffer     = deque(maxlen=9)

EAR_BASE       = None
EAR_TH         = None
EAR_MULT       = 0.75
ear_calib      = []
closed_counter = 0


# ──────────────────────────────────────────────
# Request schema
# ──────────────────────────────────────────────
class Payload(BaseModel):
    landmarks_5 : list
    true_ear    : float
    fps         : float | None = None


# ──────────────────────────────────────────────
# Inference endpoint
# ──────────────────────────────────────────────
@app.post("/infer")
def infer(data: Payload):
    global EAR_BASE, EAR_TH, closed_counter

    fps               = float(data.fps) if (data.fps is not None and data.fps > 1) else 30.0
    CALIB_FRAMES      = max(20, int(2.0 * fps))
    MICROSLEEP_FRAMES = max(3,  int(0.8 * fps))
    te                = float(data.true_ear)

    # Phase 1 — calibration
    if EAR_BASE is None:
        ear_calib.append(te)
        if len(ear_calib) >= CALIB_FRAMES:
            EAR_BASE = float(np.median(ear_calib))
            EAR_TH   = EAR_BASE * EAR_MULT
            print(f"EAR calibrated: base={EAR_BASE:.4f}  threshold={EAR_TH:.4f}")
        return {"state": "calibrating", "confidence": [0.0, 0.0, 0.0],
                "source": "CALIB",
                "debug": {"frames": len(ear_calib), "need": CALIB_FRAMES}}

    # Phase 2 — rule-based microsleep
    closed_counter = closed_counter + 1 if te < EAR_TH else max(0, closed_counter - 2)

    if closed_counter >= MICROSLEEP_FRAMES:
        pred_buffer.clear()
        return {"state": "microsleep", "confidence": [0.0, 1.0, 0.0],
                "source": "RULE",
                "debug": {"true_ear": round(te, 4), "ear_th": round(EAR_TH, 4),
                          "closed_frames": closed_counter, "needed": MICROSLEEP_FRAMES}}

    # Phase 3 — BiLSTM for alert / yawning
    features_buffer.append(compute_features_4(data.landmarks_5))

    if len(features_buffer) < SEQ_LEN:
        return {"state": "buffering", "confidence": [0.0, 0.0, 0.0],
                "source": "BUFFER",
                "debug": {"frames": len(features_buffer), "need": SEQ_LEN}}

    x = torch.tensor(
        np.array(features_buffer, dtype=np.float32)
    ).unsqueeze(0).to(device)                          # (1, 32, 4)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()

    raw_state = CLASS_NAMES[int(np.argmax(probs))]
    if raw_state == "microsleep":      # model cannot override rule
        raw_state = "alert"

    pred_buffer.append(raw_state)
    smooth_state = max(set(pred_buffer), key=pred_buffer.count)

    return {"state": smooth_state,
            "confidence": [round(float(p), 4) for p in probs],
            "source": "MODEL",
            "debug": {"true_ear": round(te, 4), "ear_th": round(EAR_TH, 4),
                      "model_raw": raw_state}}


# ──────────────────────────────────────────────
# Health check
# ──────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "device": str(device), "model": MODEL_PATH}