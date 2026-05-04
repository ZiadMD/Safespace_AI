# Notebooks

Jupyter notebooks for training, testing, and exporting SafeSpace AI models.

## Notebook Index

| Notebook | Purpose |
|----------|---------|
| `models_info.ipynb` | Model architecture analysis and comparison |
| `dataset_preparation.ipynb` | Dataset download, formatting, and validation |
| `model_training.ipynb` | YOLOv8 model training with W&B logging |
| `model_test.ipynb` | Inference testing and performance evaluation |
| `model_export.ipynb` | Model export (ONNX, TFLite, etc.) |
| `imx500_export.ipynb` | Sony IMX500 hardware model conversion |
| `ocr_script.ipynb` | License plate OCR experimentation |

## Cross-Platform Support

All notebooks now include a cross-platform setup cell. When running on:
- **Local**: Loads keys from the `.env` file in the repository root.
- **Google Colab**: Prompts/loads from Colab secrets and installs dependencies automatically.
- **Kaggle**: Loads from Kaggle Secrets and installs dependencies automatically.

## Quick Start

```bash
# Set up your .env file
cp .env.example .env
# Edit .env with your W&B and Roboflow keys

# Activate environment
source .venv/bin/activate

# Launch Jupyter
jupyter notebook notebooks/
```

## Training Config Template

```python
from ultralytics import YOLO
import torch

TRAINING_CONFIG = {
    'data': 'path/to/data.yaml',
    'epochs': 40,
    'batch': 16,
    'imgsz': 640,
    'device': 0 if torch.cuda.is_available() else 'cpu',
    'workers': 8,
    'patience': 20,
    'save': True,
    'project': '../experiments/runs',
    'name': 'experiment_name',
    'exist_ok': True,
    'pretrained': True,
    'optimizer': 'auto',
    'verbose': True,
    'seed': 42,
    'val': True,
    'plots': True,
}

model = YOLO('yolov8n.pt')
model.train(**TRAINING_CONFIG)
```

> **Note**: Training outputs go to `../experiments/runs/`. Copy the best weights to `../models/` when ready.

> **Warning**: Do not delete notebooks after training — they are valuable for reproducibility.