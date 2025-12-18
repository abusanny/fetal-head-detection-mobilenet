# Fetal Head Detection using MobileNet with Transfer Learning

## Project Overview

This repository contains a transfer learning-based approach for automated fetal head detection in ultrasound images using MobileNet architecture combined with binary focal cross-entropy loss for handling class imbalance. The model achieves robust performance on 24,036 ultrasound phantom dataset images using K-fold cross-validation with comprehensive evaluation metrics.

## Key Features

- **Architecture**: MobileNet pretrained weights with fine-tuning for binary classification (head/no-head)
- **Loss Function**: Binary Focal Cross-Entropy for addressing class imbalance
- **Preprocessing**: Image normalization, augmentation, and 224x224 cropping
- **Validation**: 5-fold cross-validation for robust model evaluation
- **Metrics**: Confusion matrix, classification reports, ROC-AUC analysis
- **Dataset**: 24,036 ultrasound phantom images with balanced preprocessing

## Dataset

- **Total Images**: 24,036 ultrasound phantom dataset samples
- **Image Size**: 224x224 pixels (resized from original dimensions)
- **Data Augmentation**: Rotation, horizontal flip, brightness/contrast adjustments
- **Class Distribution**: Balanced head/no-head detection labels
- **Train/Test Split**: K-fold cross-validation (k=5)

## Model Architecture

### Base Network
- **MobileNet V2**: Lightweight convolutional neural network designed for mobile/embedded applications
- **Pretrained Weights**: ImageNet weights for transfer learning
- **Input Shape**: (224, 224, 3)
- **Output**: Binary classification (sigmoid activation)

### Custom Layers
- Global Average Pooling
- Dense layer (128 units, ReLU activation)
- Dropout (0.5 regularization)
- Output Dense layer (1 unit, sigmoid activation)

## Training Configuration

- **Optimizer**: Adam (learning rate: 1e-4)
- **Loss Function**: Binary Focal Cross-Entropy
  - Alpha: 0.25 (focal loss parameter)
  - Gamma: 2.0 (focusing parameter)
- **Batch Size**: 32
- **Epochs**: 50 (with early stopping)
- **Early Stopping**: Patience of 10 epochs on validation loss
- **Learning Rate Reduction**: ReduceLROnPlateau (factor=0.5, patience=5)

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning Rate | 1e-4 |
| Dropout Rate | 0.5 |
| Batch Size | 32 |
| Epochs | 50 |
| Class Weights | Calculated from class distribution |
| Validation Split | 20% (5-fold cross-validation) |

## Preprocessing Pipeline

1. **Image Loading**: Load images using OpenCV
2. **Resizing**: Resize to 224x224 pixels
3. **Normalization**: Normalize pixel values to [0, 1]
4. **Augmentation** (Training only):
   - Random rotation (15 degrees)
   - Horizontal flip (50% probability)
   - Brightness adjustment (±20%)
   - Contrast adjustment (±20%)

## Model Performance

### Evaluation Metrics
- **Accuracy**: Proportion of correct predictions
- **Precision**: TP / (TP + FP) - Positive predictive value
- **Recall**: TP / (TP + FN) - Sensitivity
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve
- **Confusion Matrix**: Visualizes true positives, false positives, true negatives, false negatives

### Cross-Validation Results
Results from 5-fold cross-validation:
- **Mean Accuracy**: [Calculated from model]
- **Mean ROC-AUC**: [Calculated from model]
- **Mean F1-Score**: [Calculated from model]

## Installation

### Requirements
```bash
pip install -r requirements.txt
```

### Dependencies
- Python >= 3.8
- TensorFlow >= 2.13.0
- Keras >= 2.13.0
- NumPy >= 1.24.0
- OpenCV >= 4.8.0
- scikit-learn >= 1.3.0
- Matplotlib >= 3.8.0

## Usage

### Training the Model
```python
python src/train_mobilenet.py --data_path ./data --output_path ./models
```

### Prediction on New Images
```python
python src/predict.py --model_path ./models/best_model.h5 --image_path ./test_image.jpg
```

### Evaluation
```python
python src/evaluate.py --model_path ./models/best_model.h5 --test_data ./data/test
```

## Project Structure

```
fetal-head-detection-mobilenet/
├── src/
│   ├── train_mobilenet.py          # Training script
│   ├── predict.py                   # Prediction script
│   ├── evaluate.py                  # Evaluation metrics
│   ├── data_preprocessing.py        # Data loading and augmentation
│   └── utils.py                     # Utility functions
├── notebooks/
│   └── Training_MobileNet-1.ipynb   # Training notebook
├── data/
│   ├── train/                       # Training dataset
│   ├── val/                         # Validation dataset
│   └── test/                        # Test dataset
├── models/
│   └── best_model.h5               # Trained model weights
├── results/
│   ├── confusion_matrix.png         # Confusion matrix visualization
│   ├── roc_curve.png               # ROC curve
│   └── classification_report.txt    # Classification metrics
├── logs/
│   └── training.log                 # Training logs
├── config/
│   └── config.yaml                  # Configuration parameters
├── requirements.txt                 # Project dependencies
├── README.md                        # This file
└── PROJECT_STRUCTURE.md             # Detailed project structure
```

## Data Imbalance Handling

### Techniques Used
1. **Focal Loss**: Reduces loss contribution from easy examples, focusing on hard negatives
2. **Class Weights**: Automatically calculated to penalize misclassification of minority class
3. **Data Augmentation**: Increases effective training data by generating variations
4. **Stratified K-Fold**: Ensures balanced class distribution across folds

## Results and Analysis

### Confusion Matrix
- True Positives (TP): Correct head detections
- True Negatives (TN): Correct non-head detections
- False Positives (FP): Incorrectly detected heads
- False Negatives (FN): Missed head detections

### ROC-AUC Analysis
- Measures discrimination ability across all classification thresholds
- AUC closer to 1.0 indicates better performance
- Robust to class imbalance compared to accuracy

### Clinical Applicability
- High sensitivity (low false negatives) crucial for medical screening
- Trade-off between sensitivity and precision based on clinical requirements

## Future Improvements

1. **Model Enhancement**:
   - Explore EfficientNet and Vision Transformers
   - Implement ensemble methods with multiple architectures
   - Add attention mechanisms for spatial focus

2. **Data Expansion**:
   - Collect real clinical ultrasound data
   - Implement domain adaptation techniques
   - Increase dataset diversity

3. **Deployment**:
   - Model quantization for mobile deployment
   - TensorFlow Lite conversion
   - Real-time inference optimization

4. **Interpretability**:
   - Grad-CAM visualization for model decisions
   - Saliency maps to highlight important regions
   - SHAP analysis for feature importance

## References

- MobileNet Paper: Howard, A. G., et al. (2017). MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
- Focal Loss: Lin, T. Y., et al. (2017). Focal Loss for Dense Object Detection
- K-Fold Cross-Validation: Kohavi, R. (1995). A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection

## License

This project is provided for research and educational purposes.

## Contact

For questions or collaborations, please contact the project author.
