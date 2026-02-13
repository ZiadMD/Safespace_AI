# Safespace AI - Model Training & Testing

Safespace AI is a comprehensive AI model training and testing framework for the Safespace project. This repository handles the complete pipeline of data collection, model training, and performance evaluation using state-of-the-art YOLOv8 models.

## Project Overview

This repository serves as the central hub for:
- **Data Collection & Management**: Organizing and managing datasets for AI model training
- **Model Training**: Training YOLOv8-based models for various detection tasks (e.g., car accident detection)
- **Model Testing & Evaluation**: Comprehensive testing and validation of trained models
- **Performance Benchmarking**: Analyzing model metrics and generating evaluation reports

## Directory Structure

```
Safespace_AI/
├── Datasets/              # Dataset links and configuration
│   └── DATA_LINKS.md     # Dataset sources and documentation
├── Models/               # Trained model files and model documentation
│   ├── Car Accident.pt   # Pre-trained car accident detection model
│   └── Models.md         # Model documentation and details
├── Notebooks/            # Jupyter notebooks for training and testing
│   ├── ModelsInfo.ipynb  # Model information and analysis
│   ├── ModelTest.ipynb   # Model testing and evaluation
│   └── Notebooks.md      # Notebook usage guide
└── README.md             # This file
```

## Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **CUDA**: 11.8+ (for GPU acceleration) or CPU-only mode
- **GPU** (recommended): NVIDIA GPU with CUDA support for faster training

### Check Your CUDA Version
```bash
nvidia-smi
nvcc --version
```

If you don't have CUDA installed but have an NVIDIA GPU, follow the [official NVIDIA CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Safespace_AI
```

### 2. Create Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # For CUDA 11.8
# OR for CPU-only:
# pip install torch torchvision torchaudio
pip install ultralytics opencv-python matplotlib numpy pandas
```

### 4. Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "from ultralytics import YOLO; print('YOLOv8: Installed')"
```

## Usage

### Training a Model

Open and run the relevant notebook in the `Notebooks/` directory:

```python
from ultralytics import YOLO
import torch

# Configuration
TRAINING_CONFIG = {
    'data': r"path/to/data.yaml",                         # Path to data.yaml
    'epochs': 40,                                         # Number of training epochs
    'batch': 16,                                          # Batch size (adjust based on GPU memory)
    'imgsz': 640,                                         # Image size
    'device': 0 if torch.cuda.is_available() else 'cpu',  # GPU device or CPU
    'workers': 8,                                         # Number of data loading workers
    'patience': 20,                                       # Early stopping patience
    'save': True,                                         # Save checkpoints
    'project': 'Models',                                  # Project directory
    'name': 'Model_Name',                                 # Experiment name
    'exist_ok': True,                                     # Overwrite existing experiment
    'pretrained': True,                                   # Use pretrained weights
    'optimizer': 'auto',                                  # Optimizer (auto, SGD, Adam, AdamW, etc.)
    'verbose': True,                                      # Verbose output
    'seed': 42,                                           # Random seed for reproducibility
    'val': True,                                          # Validate during training
    'plots': True,                                        # Generate training plots
}

# Load a pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # nano, small, medium, large, xlarge available

# Train the model
results = model.train(**TRAINING_CONFIG)
```

### Testing a Model

Load and evaluate a trained model:

```python
from ultralytics import YOLO

# Load a trained model
model = YOLO('Models/Car Accident.pt')

# Run inference
results = model.predict(source='path/to/image_or_video', conf=0.25)

# Evaluate on validation set
metrics = model.val()
```

## Notebooks Overview

### [ModelsInfo.ipynb](Notebooks/ModelsInfo.ipynb)
- Model architecture and parameter analysis
- Comparison of different YOLOv8 variants
- Model information and metadata exploration

### [ModelTest.ipynb](Notebooks/ModelTest.ipynb)
- Inference testing on images and videos
- Model evaluation and performance metrics
- Visualization of predictions and results
- Metrics computation and analysis

## Datasets

Datasets are referenced in [Datasets/DATA_LINKS.md](Datasets/DATA_LINKS.md). 

To add a new dataset:
1. Add the dataset source link to `DATA_LINKS.md`
2. Document the number of classes and class names
3. Create a `data.yaml` configuration file with the dataset structure

### Dataset Format (YOLO)
```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

## Models

Trained models are stored in the [Models/](Models/) directory. 

Current Models:
- **Car Accident.pt**: YOLOv8 model trained for car accident detection

For model details, see [Models/Models.md](Models/Models.md)

## Important Notes

⚠️ **Do not delete notebooks after training** - they are valuable for future reference and reproducibility.

## Troubleshooting

### CUDA Issues
- Ensure CUDA version matches your installed PyTorch: `python -c "import torch; print(torch.version.cuda)"`
- If running on CPU, training will be slower. Adjust batch size accordingly.

### Memory Issues
- Reduce batch size if running out of GPU memory
- Reduce `imgsz` to a smaller value (e.g., 416 instead of 640)
- Reduce `workers` to 0-4 if data loading is slow

### Dataset Issues
- Ensure `data.yaml` path is correct and uses proper formatting
- Verify dataset follows YOLO format structure
- Check that images and labels are properly aligned

## Quick Start

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Open Jupyter
jupyter notebook

# 3. Run one of the notebooks:
# - Notebooks/ModelsInfo.ipynb (to explore models)
# - Notebooks/ModelTest.ipynb (to test trained models)
```

## References

- [YOLOv8 Official Documentation](https://docs.ultralytics.com)
- [PyTorch Documentation](https://pytorch.org/docs)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda)

---

**Status**: Active Development

**Last Updated**: February 2026
