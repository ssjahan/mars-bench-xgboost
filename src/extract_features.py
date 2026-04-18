"""
Extract features from frozen ViT-L/16 backbone.
This matches the paper's "feature extraction" setting but we'll use
these features for XGBoost instead of a linear classifier.
"""

import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
import numpy as np
from tqdm import tqdm
import pickle
import os
from sklearn.preprocessing import StandardScaler

class ViTFeatureExtractor:
    """
    Extract features from the penultimate layer of ViT-L/16.
    The paper uses ViT-L/16 (Section 4, Model Selection).
    """
    
    def __init__(self, model_name="google/vit-large-patch16-224", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Load ViT model WITHOUT classification head
        # We want just the transformer encoder outputs
        self.model = ViTModel.from_pretrained(model_name, use_safetensors=True)
        self.model.to(self.device)
        self.model.eval()
        
        # Get hidden size (768 for ViT-base, 1024 for ViT-large)
        self.hidden_size = self.model.config.hidden_size
        print(f"ViT hidden size: {self.hidden_size}")
        
    @torch.no_grad()
    def extract_features_from_loader(self, dataloader, description="Extracting features"):
        """
        Extract features from all images in a dataloader.
        
        Returns:
            features: numpy array of shape (n_samples, hidden_size)
            labels: numpy array of shape (n_samples,)
        """
        all_features = []
        all_labels = []
        
        for batch in tqdm(dataloader, desc=description):
            pixel_values = batch["pixel_values"].to(self.device)
            labels = batch["labels"].cpu().numpy()
            
            # Forward pass through ViT
            outputs = self.model(pixel_values)
            
            # Use [CLS] token representation (first token)
            # This is standard practice for ViT features
            cls_features = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_size)
            
            all_features.append(cls_features.cpu().numpy())
            all_labels.append(labels)
        
        features = np.vstack(all_features)
        labels = np.concatenate(all_labels)
        
        return features, labels
    
    def extract_with_pooling(self, dataloader, pool_type="mean"):
        """
        Alternative: Use mean pooling over all patch tokens.
        Sometimes works better for spatial tasks.
        """
        all_features = []
        all_labels = []
        
        for batch in tqdm(dataloader, desc=f"Extracting features ({pool_type} pool)"):
            pixel_values = batch["pixel_values"].to(self.device)
            labels = batch["labels"].cpu().numpy()
            
            outputs = self.model(pixel_values)
            # last_hidden_state: (batch, num_patches+1, hidden_size)
            # Exclude [CLS] token for mean pooling
            patch_tokens = outputs.last_hidden_state[:, 1:, :]
            
            if pool_type == "mean":
                features = patch_tokens.mean(dim=1)
            elif pool_type == "max":
                features = patch_tokens.max(dim=1)[0]
            else:
                features = outputs.last_hidden_state[:, 0, :]  # CLS token
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels)
        
        return np.vstack(all_features), np.concatenate(all_labels)


def extract_all_features(train_loader, val_loader, test_loader, save_dir="outputs/features"):
    """
    Extract features for all splits and save to disk.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    extractor = ViTFeatureExtractor()
    
    # Extract features for each split
    print("\n=== Extracting training features ===")
    X_train, y_train = extractor.extract_features_from_loader(train_loader, "Train")
    
    print("\n=== Extracting validation features ===")
    X_val, y_val = extractor.extract_features_from_loader(val_loader, "Validation")
    
    print("\n=== Extracting test features ===")
    X_test, y_test = extractor.extract_features_from_loader(test_loader, "Test")
    
    # Standardize features (important for XGBoost)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save everything
    data = {
        "X_train": X_train_scaled,
        "y_train": y_train,
        "X_val": X_val_scaled,
        "y_val": y_val,
        "X_test": X_test_scaled,
        "y_test": y_test,
        "scaler": scaler
    }
    
    with open(f"{save_dir}/vit_features.pkl", "wb") as f:
        pickle.dump(data, f)
    
    print(f"\nFeatures saved to {save_dir}/vit_features.pkl")
    print(f"Train shape: {X_train_scaled.shape}")
    print(f"Val shape: {X_val_scaled.shape}")
    print(f"Test shape: {X_test_scaled.shape}")
    
    return data


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))
    # First load data (from step 1)
    from download_data import load_mb_domars16k, prepare_dataloaders
    
    train_dataset, val_dataset, test_dataset, classes = load_mb_domars16k()
    train_loader, val_loader, test_loader, processor = prepare_dataloaders(
        train_dataset, val_dataset, test_dataset
    )
    
    # Extract features
    feature_data = extract_all_features(train_loader, val_loader, test_loader)
