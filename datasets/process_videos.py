import cv2
import os
import glob
import pandas as pd
import sys

# Add parent dir to path so we can import backend modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.vision.pose_estimator import PoseEstimator
from backend.feature_extraction.biomechanics import BiomechanicsExtractor

def process_all_videos(input_dir='datasets/raw_videos', output_csv='datasets/processed/extracted_features.csv'):
    """
    Processes all MP4 videos in the input directory.
    Extracts the peak risk frame (max valgus / min flexion) for each video.
    Saves the aggregated features to a CSV.
    """
    if not os.path.exists(input_dir):
        print(f"Directory {input_dir} not found. Please create it and add .mp4 videos.")
        return
        
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    video_files = glob.glob(os.path.join(input_dir, '*.mp4'))
    if not video_files:
        print(f"No .mp4 files found in {input_dir}.")
        
        # Create a dummy video logic if no videos exist just to show it works
        print("Generating a dummy extracted_features.csv for pipeline testing...")
        df_dummy = pd.read_csv('data/biomechanics_synthetic.csv') if os.path.exists('data/biomechanics_synthetic.csv') else None
        if df_dummy is not None:
            df_dummy.to_csv(output_csv, index=False)
            print(f"Copied synthetic data to {output_csv} as a placeholder.")
        return

    pose_estimator = PoseEstimator(static_image_mode=False)
    biomechanics = BiomechanicsExtractor()
    
    all_data = []

    print(f"Found {len(video_files)} videos to process.")

    for video_path in video_files:
        print(f"Processing: {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        best_features = None
        max_risk_score = -1 # A heuristic to find the "landing" frame
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            results = pose_estimator.process_frame(frame)
            landmarks = pose_estimator.extract_landmarks(results)
            
            if landmarks:
                features = biomechanics.extract_features(landmarks)
                if features:
                    # Heuristic for the "landing" frame: max knee flexion and valgus
                    # We want the frame where valgus is highest
                    current_risk = features['model_input']['knee_valgus_angle']
                    if current_risk > max_risk_score:
                        max_risk_score = current_risk
                        best_features = features['model_input']
        
        cap.release()
        
        if best_features:
            # We add a fake label based on thresholds if true labels aren't available
            # In reality, a physiotherapist would label this video High/Low risk.
            is_high_risk = 1 if best_features['knee_valgus_angle'] > 12.0 else 0
            
            row = {
                'video_name': os.path.basename(video_path),
                'knee_valgus_angle': best_features['knee_valgus_angle'],
                'knee_flexion_ic': best_features['knee_flexion_ic'],
                'peak_grf_bw': best_features['peak_grf_bw'],
                'hip_internal_rotation': best_features['hip_internal_rotation'],
                'trunk_lateral_flexion': best_features['trunk_lateral_flexion'],
                'injury_risk': is_high_risk
            }
            all_data.append(row)
            print(f"  -> Extracted features. Assigned Risk: {'High' if is_high_risk else 'Low'}")
        else:
            print(f"  -> No valid pose found in video.")

    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(output_csv, index=False)
        print(f"Successfully processed {len(all_data)} videos. Data saved to {output_csv}")

if __name__ == "__main__":
    process_all_videos()
