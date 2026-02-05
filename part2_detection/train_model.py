import os
import torch
import torch.nn as nn
import numpy as np
import json
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from part2 import config, model

def main():
    print("--- Part 2: Detection Model Training Pipeline (with Scaler) ---")
    
    # 1. Setup paths
    train_dir = os.path.join(config.BASE_DIR, "..", "data", "train")
    val_dir = os.path.join(config.BASE_DIR, "..", "data", "val")
    train_labels = os.path.join(train_dir, "labels.json")
    val_labels = os.path.join(val_dir, "labels.json")
    
    if not os.path.exists(train_dir) or not os.path.exists(train_labels):
        print("Data not found. Please run 'python tools/generate_data.py' first.")
        return

    # 2. Load all training data to fit scaler
    print("Loading training data...")
    with open(train_labels, "r") as f:
        train_labels_map = json.load(f)
    
    X_train = []
    y_train = []
    
    for filename, label in train_labels_map.items():
        path = os.path.join(train_dir, filename)
        if not os.path.exists(path):
            continue
        data = np.load(path, allow_pickle=True)
        embeddings = data["embeddings"]
        acoustic = json.loads(str(data["acoustic"]))
        
        # Concatenate features (same order as utils.prepare_input)
        ac_keys = sorted(acoustic.keys())
        ac_vals = np.array([acoustic[k] for k in ac_keys], dtype=np.float32)
        # AC only
        combined = ac_vals
        
        X_train.append(combined)
        y_train.append(label)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print(f"  Loaded {len(X_train)} training samples, feature dim: {X_train.shape[1]}")
    
    # 3. Load validation data
    print("Loading validation data...")
    with open(val_labels, "r") as f:
        val_labels_map = json.load(f)
    
    X_val = []
    y_val = []
    
    for filename, label in val_labels_map.items():
        path = os.path.join(val_dir, filename)
        if not os.path.exists(path):
            continue
        data = np.load(path, allow_pickle=True)
        embeddings = data["embeddings"]
        acoustic = json.loads(str(data["acoustic"]))
        
        ac_keys = sorted(acoustic.keys())
        ac_vals = np.array([acoustic[k] for k in ac_keys], dtype=np.float32)
        # AC only
        combined = ac_vals
        
        X_val.append(combined)
        y_val.append(label)
    
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    print(f"  Loaded {len(X_val)} validation samples")
    
    # 4. Fit StandardScaler on training data
    print("Fitting StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Save scaler
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, config.SCALER_PATH)
    print(f"  Scaler saved to: {config.SCALER_PATH}")
    
    # 5. Train model
    print("Training model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    clf = model.SimpleClassifier(X_train_scaled.shape[1]).to(device)
    
    # EXPERT IMPROVEMENT: Strong Label Smoothing (0.2)
    # This targets confidence in the 0.1 - 0.9 range automatically
    LABEL_SMOOTHING = 0.2
    criterion = nn.BCEWithLogitsLoss()
    
    # Increased weight decay to prevent large weights/saturation
    optimizer = torch.optim.Adam(clf.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Convert to tensors
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    
    # Smooth labels: 0 -> 0.1, 1 -> 0.9
    y_train_smooth = y_train_t * (1 - LABEL_SMOOTHING) + (0.5 * LABEL_SMOOTHING)
    
    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    
    train_ds = torch.utils.data.TensorDataset(X_train_t, y_train_smooth)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
    
    best_val_loss = float('inf')
    for epoch in range(25):
        clf.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = clf(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        clf.eval()
        with torch.no_grad():
            val_logits = clf(X_val_t.to(device))
            val_loss = criterion(val_logits, y_val_t.to(device)).item()
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_auc = roc_auc_score(y_val, val_probs)
        
        print(f"Epoch {epoch+1}/25 | Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(clf.state_dict(), config.DEFAULT_MODEL_PATH)
    
    # 6. Post-hoc Calibration (Platt Scaling)
    print("\nPerforming Platt Scaling Calibration...")
    clf.load_state_dict(torch.load(config.DEFAULT_MODEL_PATH))
    clf.eval()
    
    from part2.calibrator import TemperatureScaler
    calibrator = TemperatureScaler().to(device)
    
    with torch.no_grad():
        val_logits = clf(X_val_t.to(device))
    
    # Calibrate T and Bias on validation set
    calibrator.calibrate(val_logits, y_val_t.to(device), epochs=200)
    print(f"  Optimized Scale (1/T): {1.0/calibrator.temperature.item():.4f}, Bias: {calibrator.bias.item():.4f}")
    
    # Save Calibrator
    torch.save(calibrator.state_dict(), config.CALIBRATOR_PATH)
    
    # 7. Final evaluation
    print("\n--- Final Evaluation ---")
    with torch.no_grad():
        scaled_logits = calibrator(val_logits)
        val_probs = torch.sigmoid(scaled_logits).cpu().numpy()
        val_preds = (val_probs > 0.5).astype(int)
    
    acc = accuracy_score(y_val, val_preds)
    auc = roc_auc_score(y_val, val_probs)
    print(f"Validation Accuracy: {acc*100:.2f}%")
    print(f"Validation AUC: {auc:.4f}")
    print(f"Mean Predicted Proba (Class 1): {np.mean(val_probs):.4f}")
    
    print(f"Model saved to: {config.DEFAULT_MODEL_PATH}")
    print(f"Scaler saved to: {config.SCALER_PATH}")
    print(f"Calibrator saved to: {config.CALIBRATOR_PATH}")
    print("Training Complete.")

if __name__ == "__main__":
    main()
