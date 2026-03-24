from datasets import load_dataset
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
import random

# Where to save everything
BASE_SAVE_DIR = Path("./waste_processed")

# Final image size for all the images
TARGET_SIZE = (128, 128)

# in-distribution (ID) classes
ID_CLASSES = {"metal", "glass", "bio", "paper", "plastic"}

# Save format and quality
SAVE_EXT = ".jpg"
JPEG_QUALITY = 95

# Load dataset
ds = load_dataset("omasteam/waste-garbage-management-dataset", split="train")

label_names = None
features = ds.features
if "label" in features and hasattr(features["label"], "names"):
    label_names = features["label"].names

def get_image(row):
    return row["image"]

def preprocess_image(img):
    return img.convert("RGB").resize(TARGET_SIZE)

def map_label(label):
    label = label.lower()
    if label == "biological":
        return "bio"
    if label == "cardboard":
        return "paper"
    return label

# Output directories
id_root = BASE_SAVE_DIR / "ID"
ood_root = BASE_SAVE_DIR / "OOD"
id_root.mkdir(parents=True, exist_ok=True)
ood_root.mkdir(parents=True, exist_ok=True)

# Group dataset indices by class
id_indices_by_class = {}
ood_indices_by_class = {}

for idx, row in enumerate(ds):
    label = map_label(label_names[row["label"]])

    if label in ID_CLASSES:
        if label not in id_indices_by_class:
            id_indices_by_class[label] = []
        id_indices_by_class[label].append(idx)
    else:
        if label not in ood_indices_by_class:
            ood_indices_by_class[label] = []
        ood_indices_by_class[label].append(idx)

print("ID class counts before balancing:")
for cls in id_indices_by_class:
    count = len(id_indices_by_class[cls])
    print(cls, ":", count)

# Find smallest class size
min_id_count = None

for cls in id_indices_by_class:
    count = len(id_indices_by_class[cls])

    if min_id_count is None or count < min_id_count:
        min_id_count = count

print("\nAll ID classes will be reduced to:", min_id_count)


# Randomly select same number of samples per class
balanced_id_indices = {}

for cls in id_indices_by_class:
    indices = id_indices_by_class[cls]
    selected_indices = random.sample(indices, min_id_count)
    balanced_id_indices[cls] = selected_indices


# Save balanced ID images
id_counts = {}

for cls in balanced_id_indices:
    id_counts[cls] = 0

    class_dir = id_root / cls
    class_dir.mkdir(parents=True, exist_ok=True)

    for idx in tqdm(balanced_id_indices[cls], desc=f"Saving ID/{cls}"):
        row = ds[idx]
        img = get_image(row)
        img = preprocess_image(img)
        id_counts[cls] += 1
        filename = cls + "_" + str(id_counts[cls]).zfill(6) + SAVE_EXT
        img.save(class_dir / filename, quality=JPEG_QUALITY)

# OOD images are not balanced, we save all of them as-is
ood_counts = {}

for cls in ood_indices_by_class:
    ood_counts[cls] = 0

    class_dir = ood_root / cls
    class_dir.mkdir(parents=True, exist_ok=True)

    for idx in tqdm(ood_indices_by_class[cls], desc=f"Saving OOD/{cls}"):
        row = ds[idx]
        img = get_image(row)
        img = preprocess_image(img)
        ood_counts[cls] += 1
        filename = cls + "_" + str(ood_counts[cls]).zfill(6) + SAVE_EXT
        img.save(class_dir / filename, quality=JPEG_QUALITY)