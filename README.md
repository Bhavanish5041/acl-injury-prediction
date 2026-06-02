# Explainable AI for ACL Injury Prediction Using Vision Sensors

## Project Architecture & Implementation

This repository contains the complete implementation for real-time ACL injury prediction using vision sensors (smartphone cameras), MediaPipe pose estimation, XGBoost, and SHAP Explainable AI.

### Final Folder Structure

```
acl-injury-prediction/
├── backend/
│   ├── api/
│   │   └── main.py              # FastAPI server (WebSockets + API)
│   ├── feature_extraction/
│   │   └── biomechanics.py      # 3D joint angle math (Valgus, Flexion)
│   ├── models/                  # (Placeholder for trained XGBoost models)
│   ├── vision/
│   │   ├── camera_stream.py     # IP Webcam streaming handler
│   │   └── pose_estimator.py    # MediaPipe BlazePose wrapper
│   └── xai/
│       └── explainer.py         # SHAP logic and UI color mapping
├── datasets/
│   ├── process_videos.py        # Batch script to generate dataset from raw videos
│   ├── raw_videos/              # (Drop your jump-landing .mp4s here)
│   └── processed/               # Extracted CSV datasets
├── frontend/
│   ├── src/app/page.tsx         # Next.js Dashboard UI
│   └── (Next.js config files)
├── data/                        # Legacy/Synthetic data fallback
├── docs/                        
└── reports/                     
```

## How to Run the Project

You will need two terminal windows.

### 1. Start the Backend API (FastAPI)
Ensure you are in the project root `acl-injury-prediction/`.

Install requirements if you haven't:
```bash
pip install fastapi uvicorn mediapipe opencv-python shap xgboost pandas numpy websockets
```

Run the server:
```bash
python backend/api/main.py
```
*(The server will start on http://0.0.0.0:8000)*

### 2. Start the Frontend Dashboard (Next.js)
Open a new terminal and navigate to the `frontend/` directory.

Run the development server:
```bash
npm run dev
```
*(The UI will start on http://localhost:3000)*

---

## How to Connect Your Smartphone Camera

1. **Install IP Webcam:** Download "IP Webcam" from the Google Play Store (or EpocCam/DroidCam for iOS).
2. **Start Server on Phone:** Open the app and click "Start server".
3. **Get the IP Address:** Note the IPv4 address shown on your phone screen (e.g., `http://192.168.1.50:8080`).
4. **Connect via Dashboard:**
   - Open `http://localhost:3000` in your browser.
   - In the "Camera Source" input box, paste: `http://192.168.1.50:8080/video`
   - Click **Start Feed**.

*(Note: If you just want to test with your laptop webcam, enter `0` in the Camera Source box).*

---

## Generating Your Custom Dataset
Since clinical ACL datasets are private, use the included batch script to build your own from videos:
1. Record `.mp4` videos of jump landings on your phone.
2. Place them in `datasets/raw_videos/`.
3. Run `python datasets/process_videos.py`.
4. It will extract the peak kinematic features and save them to `datasets/processed/extracted_features.csv`.
