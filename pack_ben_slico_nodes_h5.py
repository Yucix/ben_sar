import argparse
import os
from typing import Optional, Sequence

import h5py
import numpy as np
from tqdm import tqdm

DATA_ROOT_DEFAULT = "/media/sata/xyx/BigEarthNet/dataset"
AUG_TYPES = ("orig", "hflip", "vflip", "rot180")


def build_nodes_dir_name(num_segments: int, patch_size: int, image_size: int) -> str:
    return f"aug_nodes_slico_seg{num_segments}_patch{patch_size}_img{image_size}"


def build_nodes_h5_name(num_segments: int, patch_size: int, image_size: int) -> str:
    return f"ben_slico_nodes_seg{num_segments}_patch{patch_size}_img{image_size}.h5"


def resolve_index_subdir(root: str, image_size: int, index_subdir: Optional[str] = None) -> str:
    if index_subdir:
        index_dir = os.path.join(root, index_subdir)
        if not os.path.isdir(index_dir):
            raise FileNotFoundError(f"Index directory not found: {index_dir}")
        return index_subdir

    expected_subdir = f"processed_pt_{image_size}_clean622"
    expected_dir = os.path.join(root, expected_subdir)
    if os.path.isdir(expected_dir):
        return expected_subdir

    raise FileNotFoundError(
        f"Index directory not found: {expected_dir}. "
        "You can set --index-subdir explicitly."
    )


def load_split_names(root: str, split: str, index_subdir: str):
    index_file = os.path.join(root, index_subdir, f"{split}.txt")
    if not os.path.exists(index_file):
        return None
    with open(index_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def get_nodes_dir(
    root: str,
    split: str,
    num_segments: int,
    patch_size: int,
    image_size: int,
    nodes_dir_name: Optional[str] = None,
) -> str:
    dir_name = nodes_dir_name if nodes_dir_name else build_nodes_dir_name(num_segments, patch_size, image_size)
    return os.path.join(root, split, dir_name)


def pack_split(
    h5_out,
    root,
    split,
    index_subdir,
    num_segments: int,
    patch_size: int,
    image_size: int,
    nodes_dir_name,
    compression,
    max_samples=0,
):
    names = load_split_names(root, split, index_subdir)
    if names is None:
        print(f"[Skip] Missing split index file: {split}")
        return

    if max_samples and max_samples > 0:
        names = names[: int(max_samples)]

    n_samples = len(names)
    if n_samples == 0:
        print(f"[Skip] Empty split: {split}")
        return

    nodes_dir = get_nodes_dir(
        root=root,
        split=split,
        num_segments=num_segments,
        patch_size=patch_size,
        image_size=image_size,
        nodes_dir_name=nodes_dir_name,
    )
    if not os.path.isdir(nodes_dir):
        raise FileNotFoundError(
            f"Nodes directory not found: {nodes_dir}. "
            "Please run precompute_ben_slico_nodes.py first."
        )

    # Probe first file to infer feature dimension.
    first_base = os.path.splitext(os.path.basename(names[0]))[0]
    first_path = os.path.join(nodes_dir, f"{first_base}_orig.npy")
    if not os.path.exists(first_path):
        raise FileNotFoundError(f"Missing node file: {first_path}")

    feat_dim = int(np.load(first_path).shape[1])

    grp = h5_out.require_group(split)
    index = np.zeros((n_samples, len(AUG_TYPES), 2), dtype=np.int64)
    base_names = []
    file_matrix = []

    # Phase-1: scan only (mmap header + shape), build index and total length.
    total_nodes = 0
    for i, full_name in enumerate(tqdm(names, desc=f"scan-{split}")):
        base_name = os.path.splitext(os.path.basename(full_name))[0]
        base_names.append(base_name)
        row_paths = []

        for j, aug in enumerate(AUG_TYPES):
            npy_path = os.path.join(nodes_dir, f"{base_name}_{aug}.npy")
            if not os.path.exists(npy_path):
                raise FileNotFoundError(f"Missing node file: {npy_path}")
            row_paths.append(npy_path)

            nodes_mmap = np.load(npy_path, mmap_mode="r")
            if nodes_mmap.ndim != 2:
                raise ValueError(f"Invalid node shape {nodes_mmap.shape} in {npy_path}")
            if nodes_mmap.shape[1] != feat_dim:
                raise ValueError(
                    f"Feature dim mismatch in {npy_path}. "
                    f"expected {feat_dim}, got {nodes_mmap.shape[1]}"
                )

            n = int(nodes_mmap.shape[0])
            if n <= 0:
                raise ValueError(f"Empty nodes in {npy_path}")

            index[i, j, 0] = total_nodes
            index[i, j, 1] = n
            total_nodes += n

        file_matrix.append(row_paths)

    # Phase-2: preallocate once, then sequential write.
    ds_data = grp.create_dataset(
        "data",
        shape=(total_nodes, feat_dim),
        dtype=np.float32,
        chunks=(4096, feat_dim),
        compression=compression,
    )

    for i in tqdm(range(n_samples), desc=f"write-{split}"):
        for j in range(len(AUG_TYPES)):
            offset = int(index[i, j, 0])
            length = int(index[i, j, 1])
            npy_path = file_matrix[i][j]

            nodes = np.load(npy_path).astype(np.float32, copy=False)
            ds_data[offset: offset + length] = nodes

    grp.create_dataset("index", data=index, dtype=np.int64)
    grp.create_dataset("names", data=np.asarray(base_names, dtype="S128"))

    print(
        f"[Done] split={split}, samples={n_samples}, total_nodes={total_nodes}, feat_dim={feat_dim}, "
        f"data_shape={ds_data.shape}"
    )


def pack_ben_nodes_h5(
    data_root: str,
    output_h5: str,
    image_size: int = 128,
    num_segments: int = 64,
    patch_size: int = 8,
    splits: Sequence[str] = ("train", "val"),
    index_subdir: Optional[str] = None,
    nodes_dir_name: Optional[str] = None,
    compression: Optional[str] = "lzf",
    max_samples: int = 0,
):
    out_dir = os.path.dirname(output_h5)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    resolved_index_subdir = resolve_index_subdir(
        root=data_root,
        image_size=image_size,
        index_subdir=index_subdir,
    )
    resolved_nodes_dir_name = (
        nodes_dir_name if nodes_dir_name else build_nodes_dir_name(num_segments, patch_size, image_size)
    )

    print(f"[Pack] index dir  : {resolved_index_subdir}")
    print(
        "[Pack] spec       : "
        f"seg={num_segments}, patch={patch_size}, img={image_size}"
    )
    print(f"[Pack] nodes dir  : {resolved_nodes_dir_name}")
    print(f"[Pack] output h5  : {output_h5}")

    with h5py.File(output_h5, "w") as h5_out:
        h5_out.attrs["num_segments"] = int(num_segments)
        h5_out.attrs["patch_size"] = int(patch_size)
        h5_out.attrs["image_size"] = int(image_size)
        h5_out.attrs["aug_types"] = np.asarray(AUG_TYPES, dtype="S16")

        for split in splits:
            pack_split(
                h5_out=h5_out,
                root=data_root,
                split=split,
                index_subdir=resolved_index_subdir,
                num_segments=num_segments,
                patch_size=patch_size,
                image_size=image_size,
                nodes_dir_name=nodes_dir_name,
                compression=compression,
                max_samples=max_samples,
            )

    print(f"Packed nodes saved to: {output_h5}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pack precomputed BEN SLICO node .npy files into one HDF5 file."
    )
    parser.add_argument("--data-root", type=str, default=DATA_ROOT_DEFAULT)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-segments", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument(
        "--index-subdir",
        type=str,
        default="",
        help="optional override, e.g. processed_pt_<image_size>_clean622",
    )
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val"])
    parser.add_argument(
        "--nodes-dir",
        "--nodes-dir-name",
        dest="nodes_dir",
        type=str,
        default="",
        help=(
            "optional nodes directory name under each split; "
            "default: aug_nodes_slico_seg<num_segments>_patch<patch_size>_img<image_size>"
        ),
    )
    parser.add_argument(
        "--output-h5",
        type=str,
        default="",
        help=(
            "packed output path; "
            "default: <data-root>/ben_slico_nodes_seg<num_segments>_patch<patch_size>_img<image_size>.h5"
        ),
    )
    parser.add_argument("--compression", type=str, default="none", choices=["lzf", "gzip", "none"])
    parser.add_argument("--max-samples", type=int, default=0)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    output_h5 = args.output_h5 or os.path.join(
        args.data_root,
        build_nodes_h5_name(args.num_segments, args.patch_size, args.image_size),
    )
    compression = None if args.compression == "none" else args.compression

    pack_ben_nodes_h5(
        data_root=args.data_root,
        output_h5=output_h5,
        image_size=args.image_size,
        num_segments=args.num_segments,
        patch_size=args.patch_size,
        splits=tuple(args.splits),
        index_subdir=args.index_subdir if args.index_subdir else None,
        nodes_dir_name=args.nodes_dir if args.nodes_dir else None,
        compression=compression,
        max_samples=args.max_samples,
    )
