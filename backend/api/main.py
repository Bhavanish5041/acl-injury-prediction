"""
ACL Injury Risk AI – FastAPI Backend
=====================================
Endpoints:
  POST /camera/start?source=0          start webcam (0) or IP stream URL
  POST /camera/stop                    stop camera
  WS   /ws/stream                      continuous frame + telemetry stream
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import cv2, json, asyncio, base64, os, sys
import numpy as np
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.vision.camera_stream import CameraStream
from backend.vision.pose_estimator import PoseEstimator
from backend.feature_extraction.biomechanics import BiomechanicsExtractor
from backend.xai.explainer import RiskExplainer

# ── heavy imports gated so the server still starts if packages are missing ──
try:
    import xgboost as xgb
    import pandas as pd
    _HAVE_XGB = True
except ImportError:
    _HAVE_XGB = False
    print("WARNING: xgboost/pandas not installed — using rule-based scoring.")

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="ACL Injury Risk AI API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ── Singletons ───────────────────────────────────────────────────────────────
camera: CameraStream | None = None
pose_estimator  = PoseEstimator()
biomechanics    = BiomechanicsExtractor()
risk_history = deque(maxlen=6)

# ── Model loading / training ─────────────────────────────────────────────────
xgb_model = None
explainer: RiskExplainer | None = None

MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../data/xgb_model.json')
)

if _HAVE_XGB:
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        n_estimators=200,
        max_depth=4,
        random_state=42,
        use_label_encoder=False,
    )

    if os.path.exists(MODEL_PATH):
        xgb_model.load_model(MODEL_PATH)
        print(f"✓ Loaded XGBoost model from {MODEL_PATH}")
    else:
        print("⚙ No saved model found — training fallback on synthetic data …")
        rng = np.random.default_rng(42)
        n = 1000
        y = rng.choice([0, 1], size=n, p=[0.65, 0.35])

        df_train = pd.DataFrame({
            'left_knee_flexion':  np.where(y, rng.normal(18, 6, n), rng.normal(50, 12, n)),
            'right_knee_flexion': np.where(y, rng.normal(20, 6, n), rng.normal(50, 12, n)),
            'left_knee_valgus':   np.where(y, rng.normal(13, 4, n), rng.normal(3,  2,  n)),
            'right_knee_valgus':  np.where(y, rng.normal(14, 5, n), rng.normal(3,  2,  n)),
            'left_hip_flexion':   np.where(y, rng.normal(25, 8, n), rng.normal(55, 12, n)),
            'right_hip_flexion':  np.where(y, rng.normal(27, 8, n), rng.normal(55, 12, n)),
            'knee_asymmetry':     np.where(y, rng.normal(18, 6, n), rng.normal(4,  3,  n)),
            'hip_asymmetry':      np.where(y, rng.normal(10, 4, n), rng.normal(2,  2,  n)),
            'joint_velocity':     np.where(y, rng.normal(6.0, 1.2, n), rng.normal(2.5, 0.8, n)),
        })
        # Clip to physiological ranges
        df_train = df_train.clip(lower=0)
        xgb_model.fit(df_train, y)
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        xgb_model.save_model(MODEL_PATH)
        print("✓ Fallback model trained and saved.")

    explainer = RiskExplainer(xgb_model)

# ── Risk scoring helpers ────────────────────────────────────────────────────
def _model_features(f: dict) -> dict:
    """
    Convert live camera features into the approximate scale used by the saved
    synthetic training data. The dashboard still receives the raw live values.
    """
    scaled = dict(f)
    scaled['joint_velocity'] = float(np.clip(f.get('joint_velocity', 0.0) * 30.0, 0.0, 8.0))
    return scaled


def _rule_based_risk(f: dict) -> float:
    velocity = f.get('joint_velocity', 0.0)
    left_valgus = f.get('left_knee_valgus', 0.0)
    right_valgus = f.get('right_knee_valgus', 0.0)
    knee_asymmetry = f.get('knee_asymmetry', 0.0)

    valgus_risk = min(max(left_valgus - 7.0, 0.0) * 3.0, 22.0)
    valgus_risk += min(max(right_valgus - 7.0, 0.0) * 3.0, 22.0)
    asymmetry_risk = min(max(knee_asymmetry - 10.0, 0.0) * 1.2, 18.0)

    # Static standing/squatting is not an ACL landing event. Keep the score low
    # unless alignment is clearly poor.
    if velocity < 0.06:
        return round(float(np.clip(valgus_risk + asymmetry_risk, 2.0, 18.0)), 1)

    risk = 5.0 + valgus_risk + asymmetry_risk

    # Low flexion is risky only during a moving/landing frame. Standing straight
    # should not be called a stiff landing.
    if velocity > 0.12:
        if f.get('left_knee_flexion', 90.0) < 25.0:
            risk += 12.0
        if f.get('right_knee_flexion', 90.0) < 25.0:
            risk += 12.0
        risk += min((velocity - 0.12) * 120.0, 18.0)

    return round(float(np.clip(risk, 0.0, 100.0)), 1)


def _predict_risk_percent(live_features: dict) -> tuple[float, dict]:
    model_features = _model_features(live_features)
    rule_risk = _rule_based_risk(live_features)
    velocity = live_features.get('joint_velocity', 0.0)

    if _HAVE_XGB and xgb_model is not None and velocity >= 0.10:
        df_in = pd.DataFrame([model_features])
        model_risk = float(xgb_model.predict_proba(df_in)[0][1]) * 100.0
        # The trained model is synthetic and jump-landing oriented, so it should
        # inform the demo without dominating noisy webcam frames.
        risk = (0.20 * model_risk) + (0.80 * rule_risk)
    else:
        risk = rule_risk

    risk_history.append(float(np.clip(risk, 0.0, 100.0)))
    smoothed = np.median(risk_history)

    return round(float(smoothed), 1), model_features

# ── HTTP endpoints ────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ACL Backend running"}

@app.post("/camera/start")
def start_camera(source: str = "0"):
    global camera
    if camera is not None:
        return {"status": "Camera already running"}
    try:
        src = int(source) if source.isdigit() else source
        camera = CameraStream(src=src)
        return {"status": "Camera started", "source": str(src)}
    except Exception as e:
        return {"error": str(e)}

@app.post("/camera/stop")
def stop_camera():
    global camera
    if camera is None:
        return {"status": "Camera not running"}
    camera.stop()
    camera = None
    return {"status": "Camera stopped"}

# ── WebSocket stream ──────────────────────────────────────────────────────────
@app.websocket("/ws/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Wait until a frame is available
            if camera is None or camera.frame is None:
                await asyncio.sleep(0.05)
                continue

            frame = camera.read()
            if frame is None:
                await asyncio.sleep(0.05)
                continue

            # ── Pose estimation ──────────────────────────────────────────
            results   = pose_estimator.process_frame(frame)
            landmarks = pose_estimator.extract_landmarks(results)

            telemetry = {
                "risk_score":   0.0,
                "features":     None,
                "explanations": [],
                "shap_values":  {},
            }

            annotated = frame

            if landmarks:
                feat_result = biomechanics.extract_features(landmarks)

                if feat_result:
                    fi = feat_result['model_input']
                    telemetry['features'] = fi

                    # ── Risk score ────────────────────────────────────────
                    telemetry['risk_score'], model_fi = _predict_risk_percent(fi)

                    # ── XAI explanations ──────────────────────────────────
                    if explainer is not None:
                        xai = explainer.explain_prediction(model_fi, display_features=fi)
                        telemetry['explanations'] = xai['explanations']
                        telemetry['shap_values']  = xai['shap_values']
                        highlight = xai.get('highlight_joints', {})
                    else:
                        xai = RiskExplainer.fallback_explanation(fi)
                        telemetry['explanations'] = xai['explanations']
                        telemetry['shap_values']  = xai['shap_values']
                        highlight = xai.get('highlight_joints', {})

                    # ── Draw skeleton overlay ─────────────────────────────
                    annotated = pose_estimator.draw_landmarks(
                        frame, results, highlight_joints=highlight
                    )

            # ── Encode and send ───────────────────────────────────────────
            _, buf  = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            img_b64 = base64.b64encode(buf).decode('utf-8')

            await websocket.send_text(json.dumps({
                "image":    img_b64,
                "telemetry": telemetry,
            }))
            await asyncio.sleep(0.033)   # ~30 FPS cap

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"Stream error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
