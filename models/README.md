# Models

Trained model weights for the SafeSpace project. All `.pt` files are gitignored — download or train locally.

## Available Models

| File | Task | Base Model | Notes |
|------|------|------------|-------|
| `car_accident.pt` | Accident detection | YOLOv8 | Primary accident detector |
| `accident_model.pt` | Accident detection | YOLOv8 | Alternative/lighter variant |
| `accident_classifier.pt` | Accident classification & detection | YOLOv8 | Combined classifier |

## Usage

```python
from ultralytics import YOLO

model = YOLO('models/car_accident.pt')
results = model.predict(source='path/to/image_or_video', conf=0.25)
```

## Training New Models

See [`notebooks/`](../notebooks/) for training pipelines. Trained weights are saved to `experiments/runs/` and the best checkpoint should be copied here.
