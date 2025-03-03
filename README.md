# Brain Tumor Detection

This project implements a deep learning-based system for detecting brain tumors in MRI scans using the ResNet18 architecture. The model is trained on a curated dataset of brain MRI scans and achieves high accuracy in binary classification (tumor/no-tumor).

## Features

- **Deep Learning Model**: Utilizes ResNet18 architecture with transfer learning
- **High Accuracy**: Achieves reliable tumor detection in MRI scans
- **Easy to Use**: Simple inference pipeline for medical professionals
- **GPU Optimized**: Supports GPU acceleration for faster training and inference
- **Comprehensive Dataset**: Organized collection of MRI scans for both tumor and non-tumor cases

## Project Structure

```
├── data/
│   ├── raw/         # Raw MRI scan images
│   └── processed/    # Preprocessed images
├── models/          # Saved model checkpoints
├── src/             # Source code
│   ├── data.py      # Data loading and preprocessing
│   ├── model.py     # Model architecture
│   ├── train.py     # Training script
│   ├── predict.py   # Prediction script
│   └── utils.py     # Utility functions
└── requirements.txt # Project dependencies
```

## Technical Details

### Model Architecture
- Base model: ResNet18 (pretrained)
- Custom classification head with dropout
- Binary classification output (tumor/no-tumor)

### Dataset
- Organized in binary classes (yes/no for tumor presence)
- Images preprocessed to 224x224 pixels
- Data augmentation during training
- Normalized using ImageNet statistics

### Training Process
- Transfer learning from pretrained ResNet18
- Optimizer: Adam
- Learning rate scheduling
- Early stopping to prevent overfitting
- Data augmentation for better generalization

## Setup and Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your dataset:
- Organize MRI scans in data/raw/yes (tumor) and data/raw/no (no tumor)
- Run preprocessing script

3. Train the model:
```bash
python src/train.py
```

4. Make predictions:
```bash
python src/predict.py --image_path path/to/your/image.jpg
```

## Results

The model demonstrates strong performance in tumor detection:
- High sensitivity in detecting tumors
- Low false positive rate
- Fast inference time
- Robust to various MRI scan qualities

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset contributors
- PyTorch team
- Medical imaging community

## Citation

If you use this project in your research, please cite:

```
@software{brain_tumor_detection,
  title = {Brain Tumor Detection using ResNet18},
  author = {Your Name},
  year = {2023},
  url = {https://github.com/atifkhan94/brain-tumor-detection}
}
```