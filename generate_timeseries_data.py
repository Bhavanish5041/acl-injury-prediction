import numpy as np
import pandas as pd
import os

def generate_timeseries_data(num_samples=1000, timesteps=100):
    """
    Generates synthetic 100Hz IMU data (1 second = 100 timesteps) for a jump landing.
    Features: Accel_X, Accel_Y, Accel_Z, Gyro_X, Gyro_Y, Gyro_Z
    """
    np.random.seed(42)
    os.makedirs('data', exist_ok=True)
    
    # 0 = Low Risk, 1 = High Risk
    injury_risk = np.random.choice([0, 1], size=num_samples, p=[0.75, 0.25])
    
    # Shape: (samples, timesteps, features)
    data = np.zeros((num_samples, timesteps, 6))
    
    # Time vector
    t = np.linspace(0, 1, timesteps)
    
    for i in range(num_samples):
        is_high_risk = injury_risk[i] == 1
        
        # Base landing profile (impact peak around 20-30% of the landing phase)
        impact_time = np.random.normal(0.25, 0.05)
        
        # Accelerometer Z (Vertical Ground Reaction Force proxy)
        peak_z = np.random.normal(3.5, 0.5) if is_high_risk else np.random.normal(2.0, 0.3)
        accel_z = peak_z * np.exp(-((t - impact_time)**2) / 0.01) + np.random.normal(0, 0.1, timesteps)
        
        # Gyroscope Y (Knee Flexion Velocity proxy)
        # Low risk = smooth, deep flexion. High risk = stiff landing (less rotation)
        flexion_vel = np.random.normal(2.0, 0.5) if not is_high_risk else np.random.normal(0.8, 0.3)
        gyro_y = flexion_vel * np.sin(np.pi * t) + np.random.normal(0, 0.05, timesteps)
        
        # Accelerometer X (Medial/Lateral forces - Valgus proxy)
        # High risk = higher lateral sway/forces
        sway_x = np.random.normal(1.5, 0.4) if is_high_risk else np.random.normal(0.5, 0.2)
        accel_x = sway_x * np.sin(2 * np.pi * t) * np.exp(-3*t) + np.random.normal(0, 0.1, timesteps)
        
        # Add random noise to other axes
        accel_y = np.random.normal(0, 0.2, timesteps)
        gyro_x = np.random.normal(0, 0.1, timesteps)
        gyro_z = np.random.normal(0, 0.1, timesteps)
        
        data[i] = np.column_stack((accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z))
        
    # Save arrays
    np.save('data/timeseries_features.npy', data)
    np.save('data/timeseries_labels.npy', injury_risk)
    
    print(f"Generated {num_samples} sequences of length {timesteps} with 6 features.")
    print(f"Saved to data/timeseries_features.npy and data/timeseries_labels.npy")

if __name__ == "__main__":
    generate_timeseries_data()
