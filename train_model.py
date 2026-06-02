import pandas as pd
import xgboost as xgb
import os
from sklearn.model_selection import train_test_split

def train_and_save():
    data_path = 'data/cv_biomechanics.csv'
    if not os.path.exists(data_path):
        print("Run generate_synthetic_data.py first.")
        return
        
    df = pd.read_csv(data_path)
    X = df.drop(columns=['injury_risk'])
    y = df['injury_risk']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    
    # Save the model
    os.makedirs('data', exist_ok=True)
    model.save_model('data/xgb_model.json')
    print("Trained and saved XGBoost model to data/xgb_model.json")

if __name__ == "__main__":
    train_and_save()
