import pandas as pd
import numpy as np
import os

def generate_data(num_samples=1000):
    np.random.seed(42)
    
    # 0 = Low Risk, 1 = High Risk
    # Let's say 25% of the samples are high risk
    injury_risk = np.random.choice([0, 1], size=num_samples, p=[0.75, 0.25])
    
    # Generate features based on risk class to create a realistic separability
    
    # 1. Knee Valgus Angle (degrees) - High risk > 10-15 degrees
    knee_valgus = np.where(injury_risk == 1, 
                           np.random.normal(loc=14.0, scale=3.5, size=num_samples), 
                           np.random.normal(loc=5.0, scale=2.5, size=num_samples))
    
    # 2. Knee Flexion at Initial Contact (degrees) - High risk < 20 degrees
    knee_flexion_ic = np.where(injury_risk == 1, 
                               np.random.normal(loc=15.0, scale=4.0, size=num_samples), 
                               np.random.normal(loc=28.0, scale=5.0, size=num_samples))
    
    # 3. Peak Vertical Ground Reaction Force (normalized to body weight, xBW) - High risk > 2.5
    peak_grf = np.where(injury_risk == 1, 
                        np.random.normal(loc=2.8, scale=0.4, size=num_samples), 
                        np.random.normal(loc=2.0, scale=0.3, size=num_samples))
    
    # 4. Hip Internal Rotation (degrees) - High risk > 10
    hip_internal_rot = np.where(injury_risk == 1, 
                                np.random.normal(loc=12.0, scale=4.0, size=num_samples), 
                                np.random.normal(loc=5.0, scale=3.0, size=num_samples))
    
    # 5. Trunk Lateral Flexion (degrees) - High risk > 8
    trunk_flexion = np.where(injury_risk == 1, 
                             np.random.normal(loc=9.0, scale=3.0, size=num_samples), 
                             np.random.normal(loc=4.0, scale=2.0, size=num_samples))
    
    # Add some noise features (uncorrelated with injury risk)
    age = np.random.randint(18, 30, size=num_samples)
    height_cm = np.random.normal(175, 10, size=num_samples)
    weight_kg = np.random.normal(70, 12, size=num_samples)
    
    # Compile into a DataFrame
    df = pd.DataFrame({
        'athlete_id': range(1, num_samples + 1),
        'age': age,
        'height_cm': height_cm,
        'weight_kg': weight_kg,
        'knee_valgus_angle': knee_valgus,
        'knee_flexion_ic': knee_flexion_ic,
        'peak_grf_bw': peak_grf,
        'hip_internal_rotation': hip_internal_rot,
        'trunk_lateral_flexion': trunk_flexion,
        'injury_risk': injury_risk
    })
    
    # Add some non-linear interaction noise to make it harder
    # e.g., if height is very tall and valgus is high, risk goes up
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/biomechanics_synthetic.csv', index=False)
    print(f"Generated synthetic dataset with {num_samples} samples at data/biomechanics_synthetic.csv")
    print(df['injury_risk'].value_counts())

if __name__ == "__main__":
    generate_data()
