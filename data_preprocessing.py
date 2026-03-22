from datasets import load_dataset
from collections import Counter
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import os
import re
from tqdm.auto import tqdm

# Where to save everything
BASE_SAVE_DIR = Path("./waste_processed")

# Final image size for all the images
TARGET_SIZE = (128, 128)

# in-distribution (ID) classes
ID_CLASSES = {
    "metal",
    "glass",
    "bio",
    "paper",
    "cardboard",
    "plastic",
    "clothes",
}

# Save format and quality
SAVE_EXT = ".jpg"  
JPEG_QUALITY = 95

# Load metadata from the parquet-converted branch (FASTER!!)
ds = load_dataset("omasteam/waste-garbage-management-dataset", split="train")

print(ds[0])

rows = ds
label_names = None
features = ds.features
if "label" in features and hasattr(features["label"], "names"):
    label_names = features["label"].names


# Helper functions

def get_image(row):
    return row["image"]

def preprocess_image(img):
    return img.convert("RGB").resize(TARGET_SIZE)

# Output directories
id_root = BASE_SAVE_DIR / "ID"
ood_root = BASE_SAVE_DIR / "OOD"
id_root.mkdir(parents=True, exist_ok=True)
ood_root.mkdir(parents=True, exist_ok=True)

id_counts = {}
ood_counts = {}

# Process each image and save to the appropriate directory
for row in tqdm(rows):
    label = label_names[row["label"]]
    img = preprocess_image(get_image(row))

    if label in ID_CLASSES:
        class_dir = id_root / label
        id_counts[label] = id_counts.get(label, 0) + 1
        file_index = id_counts[label]
    else:
        class_dir = ood_root / label
        ood_counts[label] = ood_counts.get(label, 0) + 1
        file_index = ood_counts[label]

    class_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{label}_{file_index:06d}{SAVE_EXT}"
    img.save(class_dir / filename, quality=JPEG_QUALITY)