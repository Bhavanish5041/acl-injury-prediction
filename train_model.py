import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def train_and_evaluate():
    # 1. Load Data
    data_path = 'data/biomechanics_synthetic.csv'
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}. Please run generate_synthetic_data.py first.")
        return
        
    df = pd.read_csv(data_path)
    print(f"Loaded dataset with {len(df)} samples.")
    
    # 2. Preprocess Data
    # Drop athlete_id as it's not a predictive feature
    X = df.drop(columns=['athlete_id', 'injury_risk'])
    y = df['injury_risk']
    
    # Split into train and test sets (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")
    
    # 3. Train Model
    # Using XGBoost classifier
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        use_label_encoder=False
    )
    
    model.fit(X_train, y_train)
    print("Model training complete.")
    
    # 4. Evaluate Model
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    }
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Save metrics
    with open('output/metrics.txt', 'w') as f:
        f.write("--- Model Evaluation Metrics ---\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("\n--- Confusion Matrix ---\n")
        f.write(f"{conf_matrix}\n")
        
    print("Metrics saved to output/metrics.txt")
    
    # 5. Explainable AI (XAI) with SHAP
    print("Generating SHAP values and plots...")
    
    # Create tree explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # SHAP Summary Plot (Beeswarm)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title('SHAP Summary Plot - Feature Importance')
    plt.tight_layout()
    plt.savefig('output/shap_summary_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved SHAP summary plot to output/shap_summary_plot.png")
    
    # SHAP Dependence Plot for the top features
    # Let's plot the top 2 features. We can deduce them from the summary or just plot the known important ones.
    top_features = ['knee_valgus_angle', 'knee_flexion_ic']
    for feature in top_features:
        plt.figure(figsize=(8, 5))
        shap.dependence_plot(feature, shap_values, X_test, show=False)
        plt.title(f'SHAP Dependence Plot - {feature}')
        plt.tight_layout()
        plt.savefig(f'output/shap_dependence_{feature}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved SHAP dependence plot for {feature}")
        
    print("Pipeline execution completed successfully.")

if __name__ == "__main__":
    train_and_evaluate()
