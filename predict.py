import os
import torch
import cv2
import numpy as np
import argparse
from PIL import Image
from torchvision import transforms
from model import load_model

def preprocess_image(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transforms
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension

def predict(model, image_path, device='cuda'):
    # Preprocess image
    image = preprocess_image(image_path)
    image = image.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        
    return predicted.item()

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Brain Tumor Detection')
    parser.add_argument('image_path', help='Path to the MRI scan image')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model_path = 'models/brain_tumor_model.pth'
    model = load_model(model_path, device=device)
    
    # Make prediction
    prediction = predict(model, args.image_path, device)
    
    # Print result
    result = 'Tumor Detected' if prediction == 1 else 'No Tumor Detected'
    print(f'\nAnalysis Result:')
    print('----------------')
    print(f'Prediction: {result}')

def predict_batch(model, image_dir, device='cuda'):
    results = []
    
    # Process all images in directory
    for img_name in os.listdir(image_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_dir, img_name)
            prediction = predict(model, img_path, device)
            results.append({
                'image': img_name,
                'prediction': 'Tumor' if prediction == 1 else 'No Tumor'
            })
    
    return results

if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model_path = 'models/brain_tumor_model.pth'
    model = load_model(model_path, device=device)
    
    # Make predictions
    image_dir = 'data'
    results = predict_batch(model, image_dir)
    
    # Print results
    print('\nPrediction Results:')
    print('----------------')
    for result in results:
        print(f"Image: {result['image']}, Prediction: {result['prediction']}")