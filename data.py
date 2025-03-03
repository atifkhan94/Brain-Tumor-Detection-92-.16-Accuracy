import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

class BrainTumorDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {'no': 0, 'yes': 1}
        
        # Load data
        self._load_dataset()
        
    def _load_dataset(self):
        # Assuming data is organized in subdirectories where each subdirectory name is the class label
        for class_name in os.listdir(self.data_dir):
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in tqdm(os.listdir(class_dir), desc=f'Loading {class_name} images'):
                    img_path = os.path.join(class_dir, img_name)
                    if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(img_path)
                        self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load and preprocess image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.long)
        return image, label

def get_data_loaders(data_dir, batch_size=32, train_split=0.8):
    # Define data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = BrainTumorDataset(data_dir, transform=transform)
    
    # Split into train and validation sets
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader

def preprocess_data(raw_data_dir, processed_data_dir):
    """Preprocess raw MRI scans and save them to processed directory"""
    os.makedirs(processed_data_dir, exist_ok=True)
    
    for class_name in os.listdir(raw_data_dir):
        class_dir = os.path.join(raw_data_dir, class_name)
        if os.path.isdir(class_dir):
            processed_class_dir = os.path.join(processed_data_dir, class_name)
            os.makedirs(processed_class_dir, exist_ok=True)
            
            for img_name in tqdm(os.listdir(class_dir), desc=f'Processing {class_name} images'):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    
                    # Read image
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                        
                    # Convert to grayscale if needed
                    if len(img.shape) == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # Apply preprocessing steps
                    img = cv2.resize(img, (224, 224))  # Resize to standard size
                    img = cv2.equalizeHist(img)  # Enhance contrast
                    
                    # Save processed image
                    processed_path = os.path.join(processed_class_dir, img_name)
                    cv2.imwrite(processed_path, img)

if __name__ == '__main__':
    raw_data_dir = 'data'
    processed_data_dir = 'data/processed'
    preprocess_data(raw_data_dir, processed_data_dir)