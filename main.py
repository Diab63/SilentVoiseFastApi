# ═══════════════════════════════════════════════════════════════════════════════
#  Silent Voice — Sign Language AI  |  FastAPI Server  v2.3
#  FIX: Mobile camera support — auto-rotation + frame normalization
#  Run: py main.py
# ═══════════════════════════════════════════════════════════════════════════════

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
import cv2
import mediapipe as mp
import tempfile
import os
import uvicorn
from typing import List

app = FastAPI(
    title="Silent Voice — Sign Language AI v2.3",
    description="Multi-sign sentence detection with mobile camera support",
    version="2.3.0"
)

# ✅ Increase upload size limit to 100MB
class LimitUploadSize(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request._body_size_limit = 100 * 1024 * 1024
        return await call_next(request)

app.add_middleware(LimitUploadSize)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model once at startup ─────────────────────────────────────────────────
print("Loading model...")
xgb_model = joblib.load("xgb_model.pkl")
le         = joblib.load("label_encoder.pkl")
print(f"Model loaded | Classes ({len(le.classes_)}): {list(le.classes_)}")
print("🔥 SILENT VOICE v2.3 IS RUNNING 🔥")

# ── MediaPipe ──────────────────────────────────────────────────────────────────
mp_hands       = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode        = True,
    max_num_hands            = 1,
    min_detection_confidence = 0.3,
    min_tracking_confidence  = 0.3,
)

# ── Tuning parameters ──────────────────────────────────────────────────────────
WINDOW_SIZE           = 10
WINDOW_STEP           = 5
SMOOTH_KERNEL         = 3
MIN_WINDOWS           = 2
CONFIDENCE_THRESHOLD  = 0.45
SAMPLE_EVERY_N_FRAMES = 2


# ── Helpers ────────────────────────────────────────────────────────────────────

def normalize_landmarks(coords: np.ndarray) -> np.ndarray:
    """Translate to wrist origin and scale by max landmark distance."""
    origin   = coords[0].copy()
    coords   = coords - origin
    max_dist = np.max(np.linalg.norm(coords, axis=1))
    if max_dist > 0:
        coords = coords / max_dist
    return coords


def fix_frame_orientation(frame: np.ndarray) -> np.ndarray:
    """
    Auto-rotate portrait mobile video frames to landscape.
    Mobile phones record in portrait (h > w), which causes MediaPipe
    to extract landmarks with wrong orientation vs training data.
    """
    h, w = frame.shape[:2]
    if h > w:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame


def extract_landmarks_from_frame(frame_bgr: np.ndarray):
    """Return normalised 21x3 landmark array, or None if no hand found."""
    # ✅ Fix mobile portrait orientation
    frame_bgr = fix_frame_orientation(frame_bgr)

    # ✅ Resize to standard size matching training data resolution
    frame_bgr = cv2.resize(frame_bgr, (640, 480))

    rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = hands_detector.process(rgb)
    if not result.multi_hand_landmarks:
        return None
    lm     = result.multi_hand_landmarks[0].landmark
    coords = np.array([[p.x, p.y, p.z] for p in lm])
    return normalize_landmarks(coords)


def classify_window(frames_landmarks: List[np.ndarray]):
    """
    Confidence-weighted vote over a window of landmark frames.
    Returns (sign_label, avg_confidence_for_winner).
    """
    sign_scores: dict[str, float] = {}
    sign_counts: dict[str, int]   = {}

    for lm in frames_landmarks:
        proba    = xgb_model.predict_proba(lm.flatten().reshape(1, -1))[0]
        pred_idx = int(np.argmax(proba))
        sign     = le.classes_[pred_idx]
        conf     = float(proba[pred_idx])
        sign_scores[sign] = sign_scores.get(sign, 0.0) + conf
        sign_counts[sign] = sign_counts.get(sign, 0)   + 1

    winner   = max(sign_scores, key=sign_scores.get)
    avg_conf = sign_scores[winner] / sign_counts[winner]
    return winner, avg_conf


def smooth_labels(labels: List[str], kernel: int) -> List[str]:
    """
    Replace each label with the majority label in a ±kernel neighbourhood.
    Suppresses single-window flicker without blurring genuine sign boundaries.
    """
    n        = len(labels)
    smoothed = []
    for i in range(n):
        neighbourhood = labels[max(0, i - kernel): min(n, i + kernel + 1)]
        majority      = max(set(neighbourhood), key=neighbourhood.count)
        smoothed.append(majority)
    return smoothed


# ── Core: multi-sign video processor ──────────────────────────────────────────

def process_video_multisign(video_path: str):
    """
    Sliding-window sign sentence detection.
    Works for both laptop and mobile camera videos.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video file.")

    # Phase 1: Extract landmarks
    all_landmarks: List[np.ndarray] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % SAMPLE_EVERY_N_FRAMES != 0:
            continue
        lm = extract_landmarks_from_frame(frame)
        if lm is not None:
            all_landmarks.append(lm)

    cap.release()

    if len(all_landmarks) < WINDOW_SIZE:
        return None, [], "Not enough hand frames detected in video"

    # Phase 2: Sliding-window classification
    window_labels: List[str]   = []
    window_confs:  List[float] = []

    for start in range(0, len(all_landmarks) - WINDOW_SIZE + 1, WINDOW_STEP):
        window      = all_landmarks[start: start + WINDOW_SIZE]
        label, conf = classify_window(window)
        window_labels.append(label)
        window_confs.append(conf)

    if not window_labels:
        return None, [], "Video too short for windowed classification"

    # Phase 3: Smooth label sequence
    smoothed_labels = smooth_labels(window_labels, SMOOTH_KERNEL)

    # Phase 4: Run-length encode → sign segments
    segments = []
    current_sign  = smoothed_labels[0]
    current_confs = [window_confs[0]]

    for label, conf in zip(smoothed_labels[1:], window_confs[1:]):
        if label == current_sign:
            current_confs.append(conf)
        else:
            segments.append({
                "sign"       : current_sign,
                "confidences": current_confs,
                "windows"    : len(current_confs),
            })
            current_sign  = label
            current_confs = [conf]

    # Flush last segment
    segments.append({
        "sign"       : current_sign,
        "confidences": current_confs,
        "windows"    : len(current_confs),
    })

    # Phase 5: Filter short / low-confidence segments
    results = []
    for seg in segments:
        avg_conf = float(np.mean(seg["confidences"]))
        if avg_conf >= CONFIDENCE_THRESHOLD and seg["windows"] >= MIN_WINDOWS:
            results.append({
                "sign"      : seg["sign"],
                "confidence": round(avg_conf, 4),
                "frames"    : seg["windows"] * WINDOW_STEP + WINDOW_SIZE,
            })

    if not results:
        return None, [], "Signs detected but confidence too low"

    sentence = " ".join(r["sign"] for r in results)
    return sentence, results, "OK"


# ── Response schema ────────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    sentence  : str
    signs     : list
    sign_count: int
    message   : str = "OK"


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/predict/video", response_model=PredictionResponse)
async def predict_video(VideoFile: UploadFile = File(...)):
    """
    Main endpoint — called by .NET backend.
    Accepts a video file, returns a full sentence of recognised signs.
    Supports both laptop and mobile camera videos.
    """
    allowed = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    ext     = os.path.splitext(VideoFile.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported format. Allowed: {allowed}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        contents = await VideoFile.read()  # ✅ async read
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        sentence, signs, message = process_video_multisign(tmp_path)
        if sentence is None:
            raise HTTPException(422, message)
        return PredictionResponse(
            sentence   = sentence,
            signs      = signs,
            sign_count = len(signs),
            message    = f"Recognised {len(signs)} sign(s)",
        )
    finally:
        os.unlink(tmp_path)


@app.post("/predict/image", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    """Single image endpoint."""
    allowed = {".jpg", ".jpeg", ".png", ".bmp"}
    ext     = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported format. Allowed: {allowed}")

    contents = await file.read()
    arr      = np.frombuffer(contents, np.uint8)
    frame    = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Cannot decode image.")

    lm = extract_landmarks_from_frame(frame)
    if lm is None:
        raise HTTPException(422, "No hand detected in image.")

    sign, conf = classify_window([lm])
    if not sign or conf < CONFIDENCE_THRESHOLD:
        raise HTTPException(422, f"Confidence too low: {conf:.2f}")

    return PredictionResponse(
        sentence   = sign,
        signs      = [{"sign": sign, "confidence": round(conf, 4), "frames": 1}],
        sign_count = 1,
        message    = "Sign recognised",
    )


class LandmarksRequest(BaseModel):
    landmarks: List[List[float]]


@app.post("/predict/landmarks", response_model=PredictionResponse)
async def predict_landmarks(body: LandmarksRequest):
    """Fastest endpoint — Flutter sends raw ML Kit landmarks."""
    if len(body.landmarks) != 21:
        raise HTTPException(400, f"Expected 21 landmarks, got {len(body.landmarks)}")

    coords     = normalize_landmarks(np.array(body.landmarks))
    sign, conf = classify_window([coords])
    if not sign or conf < CONFIDENCE_THRESHOLD:
        raise HTTPException(422, f"Confidence too low: {conf:.2f}")

    return PredictionResponse(
        sentence   = sign,
        signs      = [{"sign": sign, "confidence": round(conf, 4), "frames": 1}],
        sign_count = 1,
        message    = "Sign recognised",
    )


@app.get("/health")
def health():
    return {
        "status"  : "ok",
        "version" : "2.3 mobile-camera support",
        "classes" : list(le.classes_),
        "parameters": {
            "window_size"    : WINDOW_SIZE,
            "window_step"    : WINDOW_STEP,
            "smooth_kernel"  : SMOOTH_KERNEL,
            "min_windows"    : MIN_WINDOWS,
            "confidence_min" : CONFIDENCE_THRESHOLD,
            "sample_every_n" : SAMPLE_EVERY_N_FRAMES,
        },
    }


@app.get("/")
def root():
    return {"message": "Silent Voice AI v2.3 — /docs for Swagger UI"}


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        timeout_keep_alive=300,
        h11_max_incomplete_event_size=104_857_600  # ✅ 100MB
    )
