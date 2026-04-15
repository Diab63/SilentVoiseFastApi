# ═══════════════════════════════════════════════════════════════
# Silent Voice — Sign Language AI | Production FastAPI (Render Ready)
# ═══════════════════════════════════════════════════════════════

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

import numpy as np
import joblib
import cv2
import mediapipe as mp
import tempfile
import os
import uvicorn

# ─────────────────────────────────────────────
# App init
# ─────────────────────────────────────────────

app = FastAPI(
    title="Silent Voice AI",
    version="2.3.0",
    description="Sign language recognition API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Globals (loaded safely on startup)
# ─────────────────────────────────────────────

xgb_model = None
le = None
hands_detector = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "xgb_model.pkl")
LE_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")


# ─────────────────────────────────────────────
# MediaPipe lazy loader (IMPORTANT for Render)
# ─────────────────────────────────────────────

def get_hands():
    global hands_detector
    if hands_detector is None:
        mp_hands = mp.solutions.hands
        hands_detector = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )
    return hands_detector


# ─────────────────────────────────────────────
# Startup (load models safely)
# ─────────────────────────────────────────────

@app.on_event("startup")
def startup():
    global xgb_model, le

    print("🔄 Loading models...")

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("xgb_model.pkl not found in project")

    if not os.path.exists(LE_PATH):
        raise RuntimeError("label_encoder.pkl not found in project")

    try:
        xgb_model = joblib.load(MODEL_PATH)
        le = joblib.load(LE_PATH)
        print("✅ Models loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {str(e)}")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def normalize_landmarks(coords: np.ndarray) -> np.ndarray:
    origin = coords[0].copy()
    coords = coords - origin
    max_dist = np.max(np.linalg.norm(coords, axis=1))
    if max_dist > 0:
        coords = coords / max_dist
    return coords


def fix_frame(frame):
    h, w = frame.shape[:2]
    if h > w:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return cv2.resize(frame, (640, 480))


def extract_landmarks(frame):
    frame = fix_frame(frame)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = get_hands().process(rgb)

    if not result.multi_hand_landmarks:
        return None

    lm = result.multi_hand_landmarks[0].landmark
    coords = np.array([[p.x, p.y, p.z] for p in lm])

    return normalize_landmarks(coords)


def classify(lm):
    if xgb_model is None or le is None:
        raise HTTPException(500, "Model not loaded")

    proba = xgb_model.predict_proba(lm.reshape(1, -1))[0]
    idx = int(np.argmax(proba))
    return le.classes_[idx], float(proba[idx])


# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────

class PredictionResponse(BaseModel):
    sentence: str
    signs: list
    sign_count: int
    message: str


class LandmarksRequest(BaseModel):
    landmarks: List[List[float]]


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Silent Voice API running"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "classes": list(le.classes_) if le else []
    }


@app.post("/predict/landmarks", response_model=PredictionResponse)
def predict_landmarks(body: LandmarksRequest):

    if len(body.landmarks) != 21:
        raise HTTPException(400, "Expected 21 landmarks")

    coords = normalize_landmarks(np.array(body.landmarks))
    sign, conf = classify(coords.flatten())

    if conf < 0.45:
        raise HTTPException(422, f"Low confidence: {conf}")

    return PredictionResponse(
        sentence=sign,
        signs=[{"sign": sign, "confidence": conf, "frames": 1}],
        sign_count=1,
        message="OK"
    )

# ─────────────────────────────────────────────
# Run (Render safe)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port
    )
