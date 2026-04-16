import os
import json
import torch
import numpy as np
import rasterio
import torch.nn.functional as F
from tqdm import tqdm
import concurrent.futures

ROOT = "/media/sata/xyx/BigEarthNet/dataset"
OUT_ROOT = os.path.join(ROOT, "processed_pt_256_clean622")
TARGET_SIZE = 256

S1_ROOT = os.path.join(ROOT, "BigEarthNet-S1-v1.0")
S2_ROOT = os.path.join(ROOT, "BigEarthNet-v1.0")

# 论文风格：RGB + VV/VH
S2_BANDS = ["B02", "B03", "B04"]
S1_BANDS = ["VV", "VH"]

SPLIT_FILES = {
    "train": "train_10p_clean_622.csv",
    "val": "val_10p_clean_622.csv",
    "test": "test_10p_clean_622.csv",
}

CLASS43 = [
    'Continuous urban fabric', 'Discontinuous urban fabric', 'Industrial or commercial units',
    'Road and rail networks and associated land', 'Port areas', 'Airports', 'Mineral extraction sites',
    'Dump sites', 'Construction sites', 'Green urban areas', 'Sport and leisure facilities',
    'Non-irrigated arable land', 'Permanently irrigated land', 'Rice fields', 'Vineyards',
    'Fruit trees and berry plantations', 'Olive groves', 'Pastures', 'Annual crops associated with permanent crops',
    'Complex cultivation patterns', 'Land principally occupied by agriculture, with significant areas of natural vegetation',
    'Agro-forestry areas', 'Broad-leaved forest', 'Coniferous forest', 'Mixed forest',
    'Natural grassland', 'Moors and heathland', 'Sclerophyllous vegetation', 'Transitional woodland/shrub',
    'Beaches, dunes, sands', 'Bare rock', 'Sparsely vegetated areas', 'Burnt areas',
    'Inland marshes', 'Peatbogs', 'Salt marshes', 'Salines', 'Intertidal flats',
    'Water courses', 'Water bodies', 'Coastal lagoons', 'Estuaries', 'Sea and ocean',
]
CLASS2IDX_43 = {c: i for i, c in enumerate(CLASS43)}

LABEL_CONVERTER = {
    0: 0, 1: 0, 2: 1, 11: 2, 12: 2, 13: 2, 14: 3, 15: 3, 16: 3, 18: 3,
    17: 4, 19: 5, 20: 6, 21: 7, 22: 8, 23: 9, 24: 10, 25: 11, 31: 11,
    26: 12, 27: 12, 28: 13, 29: 14, 33: 15, 34: 15, 35: 16, 36: 16,
    38: 17, 39: 17, 40: 18, 41: 18, 42: 18,
}


def read_band_raw(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
    return arr


def load_s2_patch(s2_patch_name):
    patch_dir = os.path.join(S2_ROOT, s2_patch_name)
    bands = []
    for band in S2_BANDS:
        tif_path = os.path.join(patch_dir, f"{s2_patch_name}_{band}.tif")
        bands.append(read_band_raw(tif_path))
    return np.stack(bands, axis=0)   # [3,H,W]


def load_s1_patch(s1_patch_name):
    patch_dir = os.path.join(S1_ROOT, s1_patch_name)
    bands = []
    for band in S1_BANDS:
        tif_path = os.path.join(patch_dir, f"{s1_patch_name}_{band}.tif")
        bands.append(read_band_raw(tif_path))
    return np.stack(bands, axis=0)   # [2,H,W]


def load_label_from_s1(s1_patch_name):
    patch_dir = os.path.join(S1_ROOT, s1_patch_name)
    json_path = os.path.join(patch_dir, f"{s1_patch_name}_labels_metadata.json")

    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    indices19 = [
        LABEL_CONVERTER[CLASS2IDX_43[x]]
        for x in meta["labels"]
        if x in CLASS2IDX_43 and CLASS2IDX_43[x] in LABEL_CONVERTER
    ]

    target = torch.zeros(19, dtype=torch.float32)
    for idx in indices19:
        target[idx] = 1.0
    return target


def process_single_sample(args):
    i, s2_patch_name, s1_patch_name, save_dir = args
    save_name = f"{i:06d}_{s2_patch_name}.pt"
    save_path = os.path.join(save_dir, save_name)

    if os.path.exists(save_path):
        return save_name

    try:
        optical = load_s2_patch(s2_patch_name)  # [3,H,W]
        sar = load_s1_patch(s1_patch_name)      # [2,H,W]
        image = torch.tensor(np.concatenate([optical, sar], axis=0), dtype=torch.float32)  # [5,H,W]
        if image.shape[-2:] != (TARGET_SIZE, TARGET_SIZE):
            image = F.interpolate(
                image.unsqueeze(0),
                size=(TARGET_SIZE, TARGET_SIZE),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        label = load_label_from_s1(s1_patch_name)

        sample = {
            "image": image,
            "label": label,
            "name": s2_patch_name,
            "s1_name": s1_patch_name,
        }

        torch.save(sample, save_path)
        return save_name
    except Exception as e:
        print(f"\nError processing {s2_patch_name}: {e}")
        return None


def process_split(split):
    split_csv = os.path.join(ROOT, SPLIT_FILES[split])
    save_dir = os.path.join(OUT_ROOT, split)
    os.makedirs(save_dir, exist_ok=True)

    with open(split_csv, "r", encoding="utf-8") as f:
        pairs = [line.strip().split(",") for line in f if line.strip()]

    task_args = [(i, pair[0], pair[1], save_dir) for i, pair in enumerate(pairs)]
    index_lines = []

    num_workers = min(16, os.cpu_count() or 4)
    print(f"[{split}] using {num_workers} workers...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for result in tqdm(
            executor.map(process_single_sample, task_args),
            total=len(task_args),
            desc=f"Processing {split}"
        ):
            if result is not None:
                index_lines.append(result)

    index_file = os.path.join(OUT_ROOT, f"{split}.txt")
    with open(index_file, "w", encoding="utf-8") as f:
        for name in index_lines:
            f.write(name + "\n")

    print(f"[done] {split}: {len(index_lines)} samples -> {save_dir}")


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    for split in ["train", "val", "test"]:
        process_split(split)
    print(f"all done. saved to: {OUT_ROOT}")


if __name__ == "__main__":
    main()
