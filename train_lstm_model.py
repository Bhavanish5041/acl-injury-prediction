import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import shap
import matplotlib.pyplot as plt
import os

# Define the LSTM Model
class RiskLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=32, num_layers=2):
        super(RiskLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, (hn, cn) = self.lstm(x)
        # Take the output of the last time step
        last_out = lstm_out[:, -1, :]
        out = self.fc(last_out)
        return self.sigmoid(out)

def train_lstm():
    # 1. Load Data
    features_path = 'data/timeseries_features.npy'
    labels_path = 'data/timeseries_labels.npy'
    
    if not os.path.exists(features_path) or not os.path.exists(labels_path):
        print("Data not found. Please run generate_timeseries_data.py first.")
        return
        
    X = np.load(features_path) # shape: (1000, 100, 6)
    y = np.load(labels_path)   # shape: (1000,)
    
    # 2. Preprocess Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 3. Model Setup
    model = RiskLSTM(input_size=6, hidden_size=32, num_layers=2)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. Training Loop
    epochs = 20
    print("Training LSTM model...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}")
            
    # 5. Evaluation
    model.eval()
    with torch.no_grad():
        y_prob = model(X_test_t).numpy()
        y_pred = (y_prob >= 0.5).astype(int)
        
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    }
    
    os.makedirs('output', exist_ok=True)
    with open('output/lstm_metrics.txt', 'w') as f:
        f.write("--- LSTM Model Evaluation Metrics ---\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
            
    print("LSTM Evaluation Metrics saved to output/lstm_metrics.txt")
    
    # 6. Deep Explainable AI with SHAP
    print("Generating SHAP DeepExplainer values (this may take a moment)...")
    
    # We use a random background dataset to integrate over
    background_samples = X_train_t[np.random.choice(X_train_t.shape[0], 50, replace=False)]
    test_samples = X_test_t[:20] # Take first 20 samples to save time
    
    explainer = shap.DeepExplainer(model, background_samples)
    shap_values = explainer.shap_values(test_samples)
    
    # shap_values is a list. We take the first element (for class 0 in binary, or the only output)
    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[0]
    else:
        shap_values_to_plot = shap_values
        
    # The shape of shap_values_to_plot is (20, 100, 6). 
    # To visualize temporal importance, let's average the absolute SHAP values across all samples
    # to see which timestep and feature contributed most globally.
    
    mean_abs_shap = np.mean(np.abs(shap_values_to_plot), axis=0) # shape: (100, 6)
    
    plt.figure(figsize=(12, 6))
    features_names = ['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']
    time_axis = np.linspace(0, 1, 100)
    
    for i in range(6):
        plt.plot(time_axis, mean_abs_shap[:, i], label=features_names[i])
        
    plt.title('Temporal Feature Importance (Mean Absolute SHAP Value over time)')
    plt.xlabel('Time (seconds during jump landing)')
    plt.ylabel('Mean |SHAP Value| (Impact on Model Output)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('output/lstm_temporal_shap.png', dpi=300)
    plt.close()
    
    print("Saved Temporal SHAP plot to output/lstm_temporal_shap.png")

if __name__ == "__main__":
    train_lstm()
