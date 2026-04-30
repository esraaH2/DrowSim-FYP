"""
DrowSim — Ultra-Lightweight Raspberry Pi Client
Optimized for Pi 4 low CPU: 160x120, 10 FPS, no frame copy, skip frames.

Run:
    source ~/drowsim/myenv/bin/activate
    python client4.py
"""

import cv2
import mediapipe as mp
import numpy as np
import requests
import pygame
import time

# ──────────────────────────────────────────────
# Server IP (your laptop)
# ──────────────────────────────────────────────
SERVER_URL = "https://drowsim.fyp.systems/infer"

# ──────────────────────────────────────────────
# Settings — tune here if still slow
# ──────────────────────────────────────────────
FRAME_W        = 160    # very small = very fast for MediaPipe
FRAME_H        = 120
TARGET_FPS     = 10     # 10 FPS is enough for drowsiness detection
PROCESS_EVERY  = 2      # only run MediaPipe on every 2nd frame (skip 1)
SHOW_WINDOW    = True   # set False if you don't need the window at all

# ──────────────────────────────────────────────
# MediaPipe indices
# ──────────────────────────────────────────────
LANDMARK_5 = [33, 263, 1, 78, 308]
L_OUT, L_IN, L_UP, L_LOW = 33, 133, 159, 145
R_OUT, R_IN, R_UP, R_LOW = 263, 362, 386, 374

def compute_true_ear(lm, w, h):
    def pt(i):
        return np.array([lm[i].x * w, lm[i].y * h], dtype=np.float32)
    le = np.linalg.norm(pt(L_UP) - pt(L_LOW)) / (np.linalg.norm(pt(L_OUT) - pt(L_IN)) + 1e-6)
    re = np.linalg.norm(pt(R_UP) - pt(R_LOW)) / (np.linalg.norm(pt(R_OUT) - pt(R_IN)) + 1e-6)
    return float((le + re) / 2.0)

# ──────────────────────────────────────────────
# Audio
# ──────────────────────────────────────────────
try:
    pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=256)
    def _beep(freq, ms, vol=0.8):
        n = int(22050 * ms / 1000)
        t = np.linspace(0, ms / 1000, n, endpoint=False)
        w = (np.sin(2 * np.pi * freq * t) * vol * 32767).astype(np.int16)
        return pygame.sndarray.make_sound(w)
    SND_MICRO = _beep(1500, 600)
    SND_YAWN  = _beep(900,  350)
    AUDIO_OK  = True
except Exception as e:
    print(f"Audio off: {e}")
    AUDIO_OK = False

def play_alert(state):
    if not AUDIO_OK:
        return
    if state == "microsleep":
        SND_MICRO.play()
    elif state == "yawning":
        SND_YAWN.play()

# ──────────────────────────────────────────────
# Display — write directly on frame, NO copy
# ──────────────────────────────────────────────
def show(frame, state, ear=None):
    if not SHOW_WINDOW:
        return
    txt = state.upper()
    cv2.putText(frame, txt, (6,  21), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
    cv2.putText(frame, txt, (5,  20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    if ear is not None:
        et = f"EAR:{ear:.2f}"
        cv2.putText(frame, et,  (6,  41), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
        cv2.putText(frame, et,  (5,  40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    cv2.imshow("DrowSim", frame)
    cv2.waitKey(1)

# ──────────────────────────────────────────────
# MediaPipe — minimal config
# ──────────────────────────────────────────────
mp_mesh = mp.solutions.face_mesh
face_mesh = mp_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,          # False = much faster
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ──────────────────────────────────────────────
# Camera
# ──────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise IOError("No webcam found.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # don't queue stale frames

FRAME_DELAY = 1.0 / TARGET_FPS

print(f"Server : {SERVER_URL}")
print(f"Res    : {FRAME_W}x{FRAME_H} @ {TARGET_FPS} FPS")
print(f"Window : {'ON' if SHOW_WINDOW else 'OFF'}")
print("Press Ctrl+C to quit.\n")

# ──────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────
last_alert = False
pkt        = 0
frame_idx  = 0
last_state = "N/A"
last_ear   = None

try:
    while True:
        t0 = time.time()

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame_idx += 1

        # ── Skip frames to reduce MediaPipe load ──
        if frame_idx % PROCESS_EVERY != 0:
            # Just show last known state, don't process
            show(frame, last_state, last_ear)
            elapsed = time.time() - t0
            time.sleep(max(0, FRAME_DELAY - elapsed))
            continue

        # ── MediaPipe ──
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        if not res.multi_face_landmarks:
            last_state = "no face"
            last_ear   = None
            show(frame, "no face")
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            elapsed = time.time() - t0
            time.sleep(max(0, FRAME_DELAY - elapsed))
            continue

        lm   = res.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]
        pts5     = [[lm[i].x * w, lm[i].y * h] for i in LANDMARK_5]
        true_ear = compute_true_ear(lm, w, h)
        last_ear = true_ear

        payload = {"landmarks_5": pts5, "true_ear": true_ear, "fps": float(TARGET_FPS)}

        try:
            r      = requests.post(SERVER_URL, json=payload, timeout=1.5).json()
            state  = r.get("state", "N/A")
            conf   = r.get("confidence", [0.0, 0.0, 0.0])
            source = r.get("source", "")
            pkt   += 1
            last_state = state

            if pkt % 20 == 0 or state in ("microsleep", "yawning"):
                print(f"[{pkt:4d}] {state:12} | {source:18} | "
                      f"A:{conf[0]:.2f} M:{conf[1]:.2f} Y:{conf[2]:.2f} | EAR:{true_ear:.3f}")

            show(frame, state, true_ear)

            if state in ("microsleep", "yawning"):
                if not last_alert:
                    play_alert(state)
                last_alert = True
            else:
                last_alert = False

        except requests.exceptions.Timeout:
            print("WARNING: timeout — check laptop server is running")
            last_state = "timeout"
            show(frame, "timeout", true_ear)
            last_alert = False

        except requests.exceptions.ConnectionError:
            print(f"WARNING: no server at {SERVER_URL}")
            last_state = "no server"
            show(frame, "no server", true_ear)
            last_alert = False
            time.sleep(1.0)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        elapsed = time.time() - t0
        time.sleep(max(0, FRAME_DELAY - elapsed))

except KeyboardInterrupt:
    print(f"\nDone. Packets sent: {pkt}")
finally:
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    if AUDIO_OK:
        pygame.mixer.quit()
