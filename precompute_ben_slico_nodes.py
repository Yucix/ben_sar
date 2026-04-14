import argparse
import multiprocessing as mp
import os
from typing import Dict, Tuple

import h5py
import numpy as np
from skimage.segmentation import slic
from skimage.transform import resize
from skimage.util import img_as_float
from tqdm import tqdm

try:
    import cv2  # optional, faster resize path if available
    HAS_CV2 = True
except Exception:
    cv2 = None
    HAS_CV2 = False

AUG_TYPES = ("orig", "hflip", "vflip", "rot180")


def apply_aug(array: np.ndarray, aug_type: str) -> np.ndarray:
    if aug_type == "orig":
        return array
    if aug_type == "hflip":
        return np.fliplr(array)
    if aug_type == "vflip":
        return np.flipud(array)
    if aug_type == "rot180":
        return np.rot90(array, k=2)
    raise ValueError(f"Unsupported aug_type: {aug_type}")


def resize_patch_np(patch_2d: np.ndarray, patch_size: int) -> np.ndarray:
    if patch_2d.shape[0] == 0 or patch_2d.shape[1] == 0:
        return np.zeros((patch_size, patch_size), dtype=np.float32)

    patch_2d = patch_2d.astype(np.float32, copy=False)

    if HAS_CV2:
        return cv2.resize(
            patch_2d,
            (patch_size, patch_size),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32, copy=False)

    resized = resize(
        patch_2d,
        (patch_size, patch_size),
        order=1,
        mode="reflect",
        anti_aliasing=False,
        preserve_range=True,
    )
    return resized.astype(np.float32, copy=False)


def build_nodes_from_labels(sar_img_hw2: np.ndarray, labels: np.ndarray, patch_size: int = 16) -> np.ndarray:
    """
    sar_img_hw2: [H, W, 2]
    labels: [H, W], values in [0, num_segments-1]
    return: [N, 2*patch_size*patch_size]
    """
    nodes = []
    num_sp = int(labels.max()) + 1

    for seg_id in range(num_sp):
        mask = labels == seg_id
        if not np.any(mask):
            continue

        ys, xs = np.where(mask)
        y1, y2 = ys.min(), ys.max() + 1
        x1, x2 = xs.min(), xs.max() + 1

        crop = sar_img_hw2[y1:y2, x1:x2, :].copy()
        crop_mask = mask[y1:y2, x1:x2].astype(np.float32)

        ch_features = []
        for ch in range(crop.shape[2]):
            patch_ch = crop[:, :, ch] * crop_mask
            patch_ch = resize_patch_np(patch_ch, patch_size)
            ch_features.append(patch_ch)

        patch_chw = np.stack(ch_features, axis=0)  # [2, patch, patch]
        nodes.append(patch_chw.reshape(-1))

    if len(nodes) == 0:
        return np.zeros((1, 2 * patch_size * patch_size), dtype=np.float32)

    return np.stack(nodes, axis=0).astype(np.float32)


def build_nodes_for_all_views(sar_img_hw2: np.ndarray, num_segments: int, patch_size: int) -> Dict[str, np.ndarray]:
    # Major speedup: run SLIC only once on original image,
    # then transform labels for augmented views.
    labels_orig = slic(
        img_as_float(sar_img_hw2),
        n_segments=num_segments,
        slic_zero=True,
        start_label=0,
        channel_axis=-1,
    ).astype(np.int16)

    nodes_by_aug = {}
    for aug in AUG_TYPES:
        sar_aug = apply_aug(sar_img_hw2, aug)
        labels_aug = apply_aug(labels_orig, aug)
        nodes_by_aug[aug] = build_nodes_from_labels(
            sar_img_hw2=sar_aug,
            labels=labels_aug,
            patch_size=patch_size,
        )

    return nodes_by_aug


def load_split_names(root: str, split: str):
    index_file = os.path.join(root, "processed_pt_120_clean622", f"{split}.txt")
    if not os.path.exists(index_file):
        return None

    with open(index_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def all_outputs_exist(out_dir: str, base_name: str) -> bool:
    for aug in AUG_TYPES:
        out_path = os.path.join(out_dir, f"{base_name}_{aug}.npy")
        if not os.path.exists(out_path):
            return False
    return True


def process_one(index: int, base_name: str, images, out_dir: str, num_segments: int, patch_size: int, skip_existing: bool) -> Tuple[int, int]:
    if skip_existing and all_outputs_exist(out_dir, base_name):
        return 0, 1

    fusion = images[index]  # [5, H, W]
    sar_hw2 = np.transpose(fusion[3:5], (1, 2, 0)).astype(np.float32, copy=False)

    nodes_by_aug = build_nodes_for_all_views(sar_hw2, num_segments=num_segments, patch_size=patch_size)

    for aug in AUG_TYPES:
        out_path = os.path.join(out_dir, f"{base_name}_{aug}.npy")
        np.save(out_path, nodes_by_aug[aug])

    return 1, 0


# Worker globals for multiprocessing
_WORKER_STATE = {}


def _worker_init(h5_path: str, split: str, out_dir: str, num_segments: int, patch_size: int, skip_existing: bool):
    h5f = h5py.File(h5_path, "r")
    _WORKER_STATE["images"] = h5f[f"{split}/images"]
    _WORKER_STATE["h5f"] = h5f
    _WORKER_STATE["out_dir"] = out_dir
    _WORKER_STATE["num_segments"] = num_segments
    _WORKER_STATE["patch_size"] = patch_size
    _WORKER_STATE["skip_existing"] = skip_existing


def _worker_process(task: Tuple[int, str]) -> Tuple[int, int]:
    idx, base_name = task
    return process_one(
        index=idx,
        base_name=base_name,
        images=_WORKER_STATE["images"],
        out_dir=_WORKER_STATE["out_dir"],
        num_segments=_WORKER_STATE["num_segments"],
        patch_size=_WORKER_STATE["patch_size"],
        skip_existing=_WORKER_STATE["skip_existing"],
    )


def precompute_ben_slico_nodes(
    data_root: str,
    h5_name: str = "ben_10p_clean_622_120.h5",
    splits=("train", "val"),
    num_segments: int = 64,
    patch_size: int = 16,
    max_samples: int = 0,
    num_workers: int = 0,
    chunksize: int = 32,
    skip_existing: bool = False,
):
    h5_path = os.path.join(data_root, h5_name)
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    for split in splits:
        names = load_split_names(data_root, split)
        if names is None:
            raise FileNotFoundError(
                f"Missing index file for split '{split}': "
                f"{os.path.join(data_root, 'processed_pt_120_clean622', f'{split}.txt')}"
            )

        with h5py.File(h5_path, "r") as h5f:
            group_key = f"{split}/images"
            if group_key not in h5f:
                print(f"[Skip] Split '{split}' not found in {h5_path}")
                continue

            images = h5f[group_key]
            n_h5 = images.shape[0]
            n_txt = len(names)
            if n_h5 != n_txt:
                print(
                    f"[Warn] split={split}: h5 samples={n_h5}, txt samples={n_txt}. "
                    f"Only first {min(n_h5, n_txt)} samples will be processed."
                )

            n_samples = min(n_h5, n_txt)
            if max_samples and max_samples > 0:
                n_samples = min(n_samples, int(max_samples))

        out_dir = os.path.join(
            data_root,
            split,
            f"aug_nodes_slico_seg{num_segments}_patch{patch_size}",
        )
        os.makedirs(out_dir, exist_ok=True)

        tasks = []
        for idx in range(n_samples):
            base_name = os.path.splitext(os.path.basename(names[idx]))[0]
            tasks.append((idx, base_name))

        print("=" * 64)
        print(
            f"Split: {split} | samples: {n_samples} | out: {out_dir} | "
            f"workers: {num_workers} | skip_existing: {skip_existing}"
        )

        made = 0
        skipped = 0

        if num_workers and num_workers > 1:
            ctx = mp.get_context("spawn")
            with ctx.Pool(
                processes=num_workers,
                initializer=_worker_init,
                initargs=(h5_path, split, out_dir, num_segments, patch_size, skip_existing),
            ) as pool:
                for m, s in tqdm(
                    pool.imap_unordered(_worker_process, tasks, chunksize=chunksize),
                    total=len(tasks),
                    desc=f"{split} progress",
                ):
                    made += m
                    skipped += s
        else:
            with h5py.File(h5_path, "r") as h5f:
                images = h5f[f"{split}/images"]
                for idx, base_name in tqdm(tasks, desc=f"{split} progress"):
                    m, s = process_one(
                        index=idx,
                        base_name=base_name,
                        images=images,
                        out_dir=out_dir,
                        num_segments=num_segments,
                        patch_size=patch_size,
                        skip_existing=skip_existing,
                    )
                    made += m
                    skipped += s

        print(f"[Done] split={split}: generated={made}, skipped={skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute BEN SLICO nodes from HDF5.")
    parser.add_argument("--data-root", type=str, default="/media/sata/xyx/BigEarthNet/dataset")
    parser.add_argument("--h5-name", type=str, default="ben_10p_clean_622_120.h5")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val"])
    parser.add_argument("--num-segments", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=0, help="optional cap per split for sanity check")
    parser.add_argument("--num-workers", type=int, default=0, help="parallel workers for preprocessing")
    parser.add_argument("--chunksize", type=int, default=32, help="chunksize for multiprocessing")
    parser.add_argument("--skip-existing", action="store_true", help="skip samples with all 4 npy already present")
    args = parser.parse_args()

    precompute_ben_slico_nodes(
        data_root=args.data_root,
        h5_name=args.h5_name,
        splits=tuple(args.splits),
        num_segments=args.num_segments,
        patch_size=args.patch_size,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        chunksize=args.chunksize,
        skip_existing=args.skip_existing,
    )
    print("Precompute finished.")
