"""
Download mb-domars16k from HuggingFace Datasets
The paper releases Mars-Bench on HF: huggingface.co/collections/Mirali33/mars-bench
"""

from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, Subset
from transformers import ViTImageProcessor
import numpy as np
from tqdm import tqdm
import os

def load_mb_domars16k():
    """
    Load the mb-domars16k dataset from HuggingFace.
    
    According to the paper (Section 3.3):
    "All datasets included in Mars-Bench will be publicly released through 
    Hugging Face Datasets... Each dataset follows a standardized schema"
    """
    
    # Load dataset - this should work once Mars-Bench is released
    # If not yet available, you may need to download from Zenodo
    dataset = load_dataset("Mirali33/mb-domars16k")
    
    # The paper includes standardized train/val/test splits (Section 3.1)
    train_dataset = dataset["train"]
    val_dataset = dataset["val"]
    test_dataset = dataset["test"]
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    
    # Get class mapping
    # The paper provides mapping.json with each dataset (Section B.2.1)
    class_mapping = train_dataset.features["label"].names
    num_classes = len(class_mapping)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {class_mapping}")
    
    return train_dataset, val_dataset, test_dataset, class_mapping


def prepare_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=32):
    """
    Prepare dataloaders with proper image preprocessing.
    Using ViT's native processor as in the paper.
    """
    
    # ViT-L/16 expects 224x224 images
    processor = ViTImageProcessor.from_pretrained("google/vit-large-patch16-224")
    
# Create PyTorch datasets with transform
class MarsDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset
        
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item["image"].convert("RGB")
        label = item["label"]
        return image, label

class MarsCollate:
    def __init__(self, processor):
        self.processor = processor
        
    def __call__(self, batch):
        images, labels = zip(*batch)
        inputs = self.processor(list(images), return_tensors="pt")
        return {
            "pixel_values": inputs["pixel_values"],
            "labels": torch.tensor(labels)
        }

def prepare_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=32):
    """
    Prepare dataloaders with proper image preprocessing.
    Using ViT's native processor as in the paper.
    """
    
    # ViT-L/16 expects 224x224 images
    processor = ViTImageProcessor.from_pretrained("google/vit-large-patch16-224")
    
    collate_fn = MarsCollate(processor)
    
    train_pt = MarsDataset(train_dataset)
    val_pt = MarsDataset(val_dataset)
    test_pt = MarsDataset(test_dataset)
    
    # On Windows, num_workers > 0 often causes issues with local functions and pickling.
    # Setting to 0 runs in the main process and is much safer for stable execution.
    num_workers = 0
    
    train_loader = DataLoader(train_pt, batch_size=batch_size, shuffle=True, 
                              collate_fn=collate_fn, num_workers=num_workers)
    val_loader = DataLoader(val_pt, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=num_workers)
    test_loader = DataLoader(test_pt, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, processor


if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset, classes = load_mb_domars16k()
    train_loader, val_loader, test_loader, processor = prepare_dataloaders(
        train_dataset, val_dataset, test_dataset
    )
    print("Data loading successful!")
