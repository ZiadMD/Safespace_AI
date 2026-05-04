# Datasets

Links and references for datasets used in SafeSpace AI model training.

## Available Datasets

### License Plate Recognition
- **Directory**: `License-Plate-Recognition-11/`
- **Format**: YOLOv8 (images + labels with train/val/test splits)
- **Source**: [Roboflow](https://roboflow.com)
- See `License-Plate-Recognition-11/README.dataset.txt` for full details.

### Car Accident Detection (Zihan)
- **Format**: YOLOv8
- **Source**: Downloaded externally (see zip, gitignored)

---

## Adding a New Dataset

1. Download and place the dataset folder here
2. Ensure it follows the YOLO format:
   ```
   dataset_name/
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
3. Document the dataset in this README with source link and class info