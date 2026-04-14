import argparse
import os

import h5py
import numpy as np
from skimage.segmentation import slic
from skimage.transform import resize
from skimage.util import img_as_float
from tqdm import tqdm

AUG_TYPES = ("orig", "hflip", "vflip", "rot180")


def apply_aug(image_hwc, aug_type):
    if aug_type == "orig":
        return image_hwc
    if aug_type == "hflip":
        return np.fliplr(image_hwc)
    if aug_type == "vflip":
        return np.flipud(image_hwc)
    if aug_type == "rot180":
        return np.rot90(image_hwc, k=2)
    raise ValueError(f"Unsupported aug_type: {aug_type}")


def resize_patch_np(patch_2d, patch_size):
    if patch_2d.shape[0] == 0 or patch_2d.shape[1] == 0:
        return np.zeros((patch_size, patch_size), dtype=np.float32)

    patch_2d = patch_2d.astype(np.float32)
    resized = resize(
        patch_2d,
        (patch_size, patch_size),
        order=1,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
    )
    return resized.astype(np.float32)


def build_nodes_from_labels(sar_img_hw2, labels, patch_size=16):
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

        crop = sar_img_hw2[y1:y2, x1:x2, :].copy()  # [h, w, 2]
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


def process_and_save(sar_img_hw2, num_segments, patch_size, out_path):
    labels = slic(
        img_as_float(sar_img_hw2),
        n_segments=num_segments,
        slic_zero=True,
        start_label=0,
        channel_axis=-1,
    ).astype(np.int16)

    nodes = build_nodes_from_labels(sar_img_hw2=sar_img_hw2, labels=labels, patch_size=patch_size)
    np.save(out_path, nodes)


def load_split_names(root, split):
    index_file = os.path.join(root, "processed_pt_120_clean622", f"{split}.txt")
    if not os.path.exists(index_file):
        return None

    with open(index_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def precompute_ben_slico_nodes(
    data_root,
    h5_name="ben_10p_clean_622_120.h5",
    splits=("train", "val", "test"),
    num_segments=64,
    patch_size=16,
    max_samples=0,
):
    h5_path = os.path.join(data_root, h5_name)
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    with h5py.File(h5_path, "r") as h5f:
        for split in splits:
            group_key = f"{split}/images"
            if group_key not in h5f:
                print(f"[Skip] Split '{split}' not found in {h5_path}")
                continue

            names = load_split_names(data_root, split)
            if names is None:
                raise FileNotFoundError(
                    f"Missing index file for split '{split}': "
                    f"{os.path.join(data_root, 'processed_pt_120_clean622', f'{split}.txt')}"
                )

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

            print("=" * 64)
            print(f"Split: {split} | samples: {n_samples} | out: {out_dir}")

            for idx in tqdm(range(n_samples), desc=f"{split} progress"):
                fusion = images[idx]  # [5, H, W]
                sar_hw2 = np.transpose(fusion[3:5], (1, 2, 0)).astype(np.float32)

                base_name = os.path.splitext(os.path.basename(names[idx]))[0]
                for aug_type in AUG_TYPES:
                    sar_aug = apply_aug(sar_hw2, aug_type)
                    out_name = f"{base_name}_{aug_type}.npy"
                    out_path = os.path.join(out_dir, out_name)
                    process_and_save(
                        sar_img_hw2=sar_aug,
                        num_segments=num_segments,
                        patch_size=patch_size,
                        out_path=out_path,
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute BEN SLICO nodes from HDF5.")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--h5-name", type=str, default="ben_10p_clean_622_120.h5")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val", "test"])
    parser.add_argument("--num-segments", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="optional cap per split for sanity check")
    args = parser.parse_args()

    precompute_ben_slico_nodes(
        data_root=args.data_root,
        h5_name=args.h5_name,
        splits=tuple(args.splits),
        num_segments=args.num_segments,
        patch_size=args.patch_size,
        max_samples=args.max_samples,
    )
    print("Precompute finished.")
