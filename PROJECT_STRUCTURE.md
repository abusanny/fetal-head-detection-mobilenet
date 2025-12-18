# Project Structure Documentation

## Overview

This document provides a detailed explanation of the project directory structure for the Fetal Head Detection using MobileNet project.

## Directory Tree

```
fetal-head-detection-mobilenet/
├── src/                                # Source code modules
│   ├── __init__.py                     # Package initialization
│   ├── train_mobilenet.py              # Training script with MobileNet
│   ├── predict.py                       # Inference/prediction module
│   ├── evaluate.py                      # Model evaluation metrics
│   ├── data_preprocessing.py            # Data loading and augmentation
│   └── utils.py                         # Utility functions
├── notebooks/                         # Jupyter notebooks
│   └── Training_MobileNet-1.ipynb       # Original training notebook
├── data/                              # Dataset directory
│   ├── train/                           # Training dataset (18,000 images)
│   │   ├── head/                        # Positive class samples
│   │   └── no_head/                     # Negative class samples
│   ├── val/                             # Validation dataset (3,000 images)
│   │   ├── head/
│   │   └── no_head/
│   └── test/                            # Test dataset (3,036 images)
│       ├── head/
│       └── no_head/
├── models/                            # Trained model artifacts
│   ├── best_model.h5                   # Best trained model weights
│   ├── model_architecture.json          # Model architecture definition
│   └── training_history.pkl            # Training history (loss, accuracy)
├── results/                           # Evaluation results and visualizations
│   ├── confusion_matrix.png             # Confusion matrix visualization
│   ├── confusion_matrix_normalized.png  # Normalized confusion matrix
│   ├── roc_curve.png                    # ROC curve plot
│   ├── classification_report.txt       # Detailed classification metrics
│   ├── performance_metrics.csv         # Cross-validation results
│   └── predictions.csv                  # Model predictions on test set
├── logs/                              # Training and execution logs
│   ├── training.log                     # Training process logs
│   ├── inference.log                    # Inference execution logs
│   └── errors.log                       # Error logs
├── config/                            # Configuration files
│   ├── config.yaml                      # Main configuration file
│   ├── model_config.yaml               # Model-specific parameters
│   └── data_config.yaml                # Data processing parameters
├── requirements.txt                 # Python package dependencies
├── README.md                        # Project readme
├── PROJECT_STRUCTURE.md             # This file
└── .gitignore                       # Git ignore file
```

## Detailed Component Descriptions

### `/src/` - Source Code

#### `train_mobilenet.py`
- **Purpose**: Main training script for the MobileNet model
- **Key Functions**:
  - `load_data()`: Load and preprocess images from directories
  - `create_model()`: Initialize MobileNet with custom layers
  - `compile_model()`: Configure optimizer, loss, and metrics
  - `train_model()`: Execute training loop with K-fold validation
  - `save_model()`: Save trained weights and architecture
- **Input**: Image data from `/data/train/`
- **Output**: Trained model weights to `/models/best_model.h5`

#### `predict.py`
- **Purpose**: Inference module for making predictions on new images
- **Key Functions**:
  - `load_trained_model()`: Load pretrained weights
  - `preprocess_image()`: Prepare single image for prediction
  - `predict()`: Generate predictions with confidence scores
  - `batch_predict()`: Process multiple images efficiently
- **Input**: Image file or directory path
- **Output**: Prediction labels and probability scores

#### `evaluate.py`
- **Purpose**: Model evaluation and metrics computation
- **Key Functions**:
  - `evaluate_model()`: Compute accuracy, precision, recall, F1
  - `confusion_matrix()`: Generate and visualize confusion matrix
  - `roc_auc_score()`: Calculate ROC-AUC metric
  - `cross_validate()`: Perform K-fold cross-validation
  - `generate_report()`: Create comprehensive evaluation report
- **Input**: Model and test data
- **Output**: Metrics and visualizations to `/results/`

#### `data_preprocessing.py`
- **Purpose**: Data loading and augmentation pipeline
- **Key Functions**:
  - `load_images()`: Read images from directory structure
  - `resize_images()`: Resize to 224x224 pixels
  - `normalize_data()`: Normalize pixel values [0-1]
  - `augment_data()`: Apply transformations (rotation, flip, brightness)
  - `create_splits()`: Generate train/val/test splits
- **Output**: Preprocessed numpy arrays or tf.data.Dataset

#### `utils.py`
- **Purpose**: Utility functions used across modules
- **Functions**:
  - `create_directories()`: Initialize required directories
  - `load_config()`: Load YAML configuration files
  - `save_config()`: Save configuration to YAML
  - `setup_logging()`: Configure logging system
  - `plot_training_history()`: Visualize training curves
  - `display_images()`: Debug image visualization

### `/notebooks/` - Jupyter Notebooks

#### `Training_MobileNet-1.ipynb`
- **Purpose**: Exploratory and experimental training notebook
- **Cells**:
  1. Environment setup and imports
  2. Data loading and EDA
  3. Model architecture definition
  4. Training loop execution
  5. Results visualization and analysis
- **Use Case**: Interactive development and experimentation

### `/data/` - Dataset

- **Total Size**: ~24 GB (24,036 images)
- **Train Set**: 75% (18,000 images)
  - `head/`: Positive class (ultrasound with fetal head)
  - `no_head/`: Negative class (ultrasound without fetal head)
- **Validation Set**: 12.5% (3,000 images)
  - Same class structure as training
- **Test Set**: 12.5% (3,036 images)
  - Used for final model evaluation
  - Kept separate from training/validation

### `/models/` - Model Artifacts

#### `best_model.h5`
- **Type**: Keras/TensorFlow HDF5 format
- **Size**: ~50-100 MB
- **Contains**: Model weights and architecture
- **Usage**: Load for inference

#### `model_architecture.json`
- **Type**: JSON serialization
- **Purpose**: Model architecture without weights
- **Use**: Reference and deployment

#### `training_history.pkl`
- **Type**: Python pickle object
- **Contents**: Training/validation loss and accuracy per epoch
- **Use**: Generate training curves and analysis

### `/results/` - Evaluation Results

#### Visualizations
- `confusion_matrix.png`: Unnormalized confusion matrix heatmap
- `confusion_matrix_normalized.png`: Row-normalized confusion matrix
- `roc_curve.png`: ROC curve with AUC score

#### Reports
- `classification_report.txt`: Precision, recall, F1-score per class
- `performance_metrics.csv`: Metrics from each K-fold iteration
- `predictions.csv`: Model predictions with confidence scores

### `/logs/` - Execution Logs

- **training.log**: Epoch-wise training metrics and system info
- **inference.log**: Prediction requests and execution times
- **errors.log**: Exceptions and warnings

### `/config/` - Configuration Files

#### `config.yaml`
```yaml
# General configuration
project_name: fetal-head-detection-mobilenet
dataset_path: ./data
model_save_path: ./models/best_model.h5
results_path: ./results
logs_path: ./logs
```

#### `model_config.yaml`
```yaml
# Model hyperparameters
learning_rate: 1e-4
batch_size: 32
epochs: 50
early_stopping_patience: 10
dropout_rate: 0.5
```

#### `data_config.yaml`
```yaml
# Data preprocessing
image_size: [224, 224]
normalization: [0, 1]
augmentation:
  rotation: 15
  flip_probability: 0.5
validation_split: 0.2
```

## File Naming Conventions

- **Python files**: `snake_case.py`
- **Notebooks**: `Descriptive_Title.ipynb`
- **Data directories**: `lowercase/`
- **Model files**: `{model_name}_{version}.h5`
- **Results**: `{metric_name}_{timestamp}.png`

## Data Flow

```
/data/raw_images
       ↓
[data_preprocessing.py]
       ↓
Normalized & Augmented Data
       ↓
[train_mobilenet.py]
       ↓
/models/best_model.h5
       ↓
[evaluate.py]
       ↓
/results/{metrics, visualizations}
```

## Usage Examples

### Training
```bash
cd src
python train_mobilenet.py --config ../config/config.yaml
```

### Prediction
```bash
cd src
python predict.py --model ../models/best_model.h5 --image_path ../data/test/image.jpg
```

### Evaluation
```bash
cd src
python evaluate.py --model ../models/best_model.h5 --test_dir ../data/test
```

## Size Estimates

| Directory | Size | Notes |
|-----------|------|-------|
| `/data/` | ~24 GB | Raw image dataset |
| `/models/` | ~100 MB | Trained weights |
| `/results/` | ~50 MB | Visualizations and reports |
| `/logs/` | ~10 MB | Training and inference logs |
| **Total** | **~24 GB** | Dominated by dataset |

## Important Notes

1. **Data Privacy**: Dataset should not be committed to repository
2. **Model Size**: Models directory can be large; consider DVC for versioning
3. **Reproducibility**: Use fixed random seeds in config files
4. **Logging**: Always enable logging for debugging and monitoring
5. **Documentation**: Keep this file updated when adding new components
