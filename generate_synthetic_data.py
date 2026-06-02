import pandas as pd
import numpy as np
import os

def generate_data(num_samples=2000):
    np.random.seed(42)
    
    # 0 = Low Risk, 1 = High Risk
    injury_risk = np.random.choice([0, 1], size=num_samples, p=[0.7, 0.3])
    
    # Features (Simulating realistic biomechanical angles from CV)
    
    # Knee Flexion (High risk = stiff landing < 25 degrees)
    left_knee_flexion = np.where(injury_risk == 1, np.random.normal(20, 5, num_samples), np.random.normal(45, 8, num_samples))
    right_knee_flexion = np.where(injury_risk == 1, np.random.normal(22, 6, num_samples), np.random.normal(44, 8, num_samples))
    
    # Knee Valgus Deviation from straight (High risk = high deviation > 10 degrees)
    left_knee_valgus = np.where(injury_risk == 1, np.random.normal(12, 4, num_samples), np.random.normal(3, 2, num_samples))
    right_knee_valgus = np.where(injury_risk == 1, np.random.normal(15, 5, num_samples), np.random.normal(4, 2, num_samples))
    
    # Hip Flexion
    left_hip_flexion = np.where(injury_risk == 1, np.random.normal(30, 10, num_samples), np.random.normal(50, 15, num_samples))
    right_hip_flexion = np.where(injury_risk == 1, np.random.normal(32, 10, num_samples), np.random.normal(52, 15, num_samples))
    
    # Asymmetry Features
    knee_asymmetry = np.abs(left_knee_flexion - right_knee_flexion) + np.abs(left_knee_valgus - right_knee_valgus)
    hip_asymmetry = np.abs(left_hip_flexion - right_hip_flexion)
    
    # Joint Velocity (Approximation of impact speed - higher is riskier)
    joint_velocity = np.where(injury_risk == 1, np.random.normal(5.5, 1.2, num_samples), np.random.normal(3.0, 0.8, num_samples))
    
    df = pd.DataFrame({
        'left_knee_flexion': left_knee_flexion,
        'right_knee_flexion': right_knee_flexion,
        'left_knee_valgus': left_knee_valgus,
        'right_knee_valgus': right_knee_valgus,
        'left_hip_flexion': left_hip_flexion,
        'right_hip_flexion': right_hip_flexion,
        'knee_asymmetry': knee_asymmetry,
        'hip_asymmetry': hip_asymmetry,
        'joint_velocity': joint_velocity,
        'injury_risk': injury_risk
    })
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/cv_biomechanics.csv', index=False)
    print(f"Generated vision-based synthetic dataset at data/cv_biomechanics.csv")

if __name__ == "__main__":
    generate_data()
