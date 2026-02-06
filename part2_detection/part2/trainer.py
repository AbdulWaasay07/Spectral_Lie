import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from . import model, config, utils

def train_model(train_loader, val_loader, epochs=20, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Init model
    # Note: verify input dim from loader first batch
    feature_dim = config.INPUT_DIM_DEFAULT # Or derive dynamically
    clf = model.SimpleClassifier(feature_dim).to(device)
    
    optimizer = optim.AdamW(clf.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    best_auc = 0.0
    
    for epoch in range(epochs):
        # Train
        clf.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = clf(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validate
        clf.eval()
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = clf(x)
                all_logits.extend(logits.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                
        # Metric
        try:
            auc = roc_auc_score(all_labels, all_logits)
        except ValueError:
            auc = 0.5 # Handle single class edge case
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | Val AUC: {auc:.4f}")
        
        # Save best
        if auc > best_auc:
            best_auc = auc
            torch.save(clf.state_dict(), config.DEFAULT_MODEL_PATH)
            print("  --> Saved new best model")
            
    return clf
