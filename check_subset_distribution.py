import os
import torch
import random
from collections import Counter

ROOT = "/media/sata/xyx/BigEarthNet/dataset"
SPLIT = "val"   # 改成 train / val 都可以测
MAX_SAMPLES = 1000
SEED = 3407

CLASS_NAMES = [
    'Urban fabric',
    'Industrial or commercial units',
    'Arable land',
    'Permanent crops',
    'Pastures',
    'Complex cultivation patterns',
    'Land principally occupied by agriculture, with significant areas of natural vegetation',
    'Agro-forestry areas',
    'Broad-leaved forest',
    'Coniferous forest',
    'Mixed forest',
    'Natural grassland and sparsely vegetated areas',
    'Moors, heathland and sclerophyllous vegetation',
    'Transitional woodland, shrub',
    'Beaches, dunes, sands',
    'Inland wetlands',
    'Coastal wetlands',
    'Inland waters',
    'Marine waters',
]

index_file = os.path.join(ROOT, "processed_pt_256", f"{SPLIT}.txt")
data_root = os.path.join(ROOT, "processed_pt_256", SPLIT)

with open(index_file, "r", encoding="utf-8") as f:
    files = [line.strip() for line in f if line.strip()]

rng = random.Random(SEED)
if MAX_SAMPLES > 0 and MAX_SAMPLES < len(files):
    files = rng.sample(files, MAX_SAMPLES)

label_count = torch.zeros(19, dtype=torch.long)
label_cardinality = []

for fn in files:
    pt_path = os.path.join(data_root, fn)
    sample = torch.load(pt_path, map_location="cpu", weights_only=True)
    y = sample["label"].long()
    label_count += y
    label_cardinality.append(int(y.sum().item()))

print(f"split={SPLIT}, samples={len(files)}")
print(f"avg labels per sample = {sum(label_cardinality) / len(label_cardinality):.4f}")
print("-" * 60)

for i, cnt in enumerate(label_count.tolist()):
    print(f"{i:02d} | {CLASS_NAMES[i]} | positives = {cnt}")