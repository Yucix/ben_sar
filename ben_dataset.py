import os
import pickle
import random
from collections import OrderedDict

import h5py
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF

BAND_MEAN = torch.tensor([
    429.9430203,              # B02
    614.21682446,             # B03
    590.23569706,             # B04
    -12.619993741972035,      # VV
    -19.29044597721542,       # VH
], dtype=torch.float32).view(5, 1, 1)

BAND_STD = torch.tensor([
    572.41639287,             # B02
    582.87945694,             # B03
    675.88746967,             # B04
    5.115911777546365,        # VV
    5.464428464912864,        # VH
], dtype=torch.float32).view(5, 1, 1)


class BEN10Dataset(data.Dataset):
    def __init__(
        self,
        root,
        split="train",
        transform=None,
        inp_name=None,
        image_size=120,
        max_samples=None,
        num_segments=64,
        patch_size=16,
        nodes_dir=None,
        nodes_backend="auto",
        nodes_h5_path=None,
        nodes_cache_budget_mb=0,
        nodes_cache_allow_workers=False,
    ):
        self.root = root
        self.split = split
        self.transform = transform
        self.image_size = image_size

        self.h5_path = os.path.join(root, "ben_10p_clean_622_120.h5")
        self.h5_file = None

        self.index_file = os.path.join(root, "processed_pt_120_clean622", f"{split}.txt")
        with open(self.index_file, "r", encoding="utf-8") as f:
            self.files = [line.strip() for line in f if line.strip()]

        if max_samples is not None and max_samples > 0 and max_samples < len(self.files):
            rng = random.Random(3407)
            self.valid_indices = rng.sample(range(len(self.files)), max_samples)
            self.files = [self.files[i] for i in self.valid_indices]
        else:
            self.valid_indices = list(range(len(self.files)))

        self.num_classes = 19

        if not inp_name or not os.path.exists(inp_name):
            raise FileNotFoundError(
                f"Embedding file not found: {inp_name}. "
                "Please generate/check `bigearthnet19_glove_word2vec.pkl`."
            )

        with open(inp_name, "rb") as f:
            self.inp = torch.tensor(pickle.load(f), dtype=torch.float32)

        if nodes_dir is None:
            self.nodes_dir = os.path.join(
                root,
                split,
                f"aug_nodes_slico_seg{num_segments}_patch{patch_size}",
            )
        else:
            self.nodes_dir = nodes_dir

        self.aug_types = ("orig", "hflip", "vflip", "rot180")
        self.aug_to_idx = {aug: i for i, aug in enumerate(self.aug_types)}

        default_nodes_h5 = os.path.join(
            root, f"ben_slico_nodes_seg{num_segments}_patch{patch_size}.h5"
        )
        self.nodes_h5_path = nodes_h5_path if nodes_h5_path else default_nodes_h5
        self.nodes_h5_file = None
        self.nodes_index = None
        self.nodes_data = None

        if nodes_backend == "auto":
            self.nodes_backend = "h5" if os.path.exists(self.nodes_h5_path) else "npy"
        else:
            self.nodes_backend = nodes_backend

        if self.nodes_backend not in ("npy", "h5"):
            raise ValueError(f"Unsupported nodes_backend={self.nodes_backend}. Choose from npy/h5/auto.")

        if self.nodes_backend == "h5" and not os.path.exists(self.nodes_h5_path):
            raise FileNotFoundError(
                f"nodes_h5 file not found: {self.nodes_h5_path}. "
                "Please generate it with pack_ben_slico_nodes_h5.py."
            )

        # Budgeted LRU cache for nodes (in-memory, hard cap with eviction).
        # By default this cache is disabled in DataLoader workers to avoid memory blow-up.
        self.nodes_cache_budget_bytes = int(max(0, nodes_cache_budget_mb) * 1024 * 1024)
        self.nodes_cache_allow_workers = bool(nodes_cache_allow_workers)
        self.nodes_cache = OrderedDict()
        self.nodes_cache_bytes = 0
        self._nodes_cache_runtime_enabled = None

        print(f"[BEN10Dataset-HDF5] {split}: {len(self.valid_indices)} samples loaded.")
        print(f"[BEN10Dataset-HDF5] {split}: nodes dir = {self.nodes_dir}")
        print(f"[BEN10Dataset-HDF5] {split}: nodes backend = {self.nodes_backend}")
        if self.nodes_backend == "h5":
            print(f"[BEN10Dataset-HDF5] {split}: nodes h5 = {self.nodes_h5_path}")
        print(
            f"[BEN10Dataset-HDF5] {split}: nodes cache budget = "
            f"{self.nodes_cache_budget_bytes / 1024 / 1024:.1f} MB "
            f"(allow_workers={self.nodes_cache_allow_workers})"
        )

    def _is_nodes_cache_enabled(self):
        if self._nodes_cache_runtime_enabled is not None:
            return self._nodes_cache_runtime_enabled

        if self.nodes_cache_budget_bytes <= 0:
            self._nodes_cache_runtime_enabled = False
            return False

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and not self.nodes_cache_allow_workers:
            self._nodes_cache_runtime_enabled = False
            return False

        self._nodes_cache_runtime_enabled = True
        return True

    def _cache_get(self, key):
        arr = self.nodes_cache.get(key, None)
        if arr is None:
            return None
        self.nodes_cache.move_to_end(key, last=True)
        return arr

    def _cache_put(self, key, arr):
        if not self._is_nodes_cache_enabled():
            return
        if arr is None:
            return

        arr_nbytes = int(arr.nbytes)
        if arr_nbytes > self.nodes_cache_budget_bytes:
            return

        old = self.nodes_cache.pop(key, None)
        if old is not None:
            self.nodes_cache_bytes -= int(old.nbytes)

        while (
            self.nodes_cache
            and self.nodes_cache_bytes + arr_nbytes > self.nodes_cache_budget_bytes
        ):
            _, evicted = self.nodes_cache.popitem(last=False)
            self.nodes_cache_bytes -= int(evicted.nbytes)

        self.nodes_cache[key] = arr
        self.nodes_cache_bytes += arr_nbytes

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")

        real_idx = self.valid_indices[index]

        fusion_np = self.h5_file[f"{self.split}/images"][real_idx]
        target_np = self.h5_file[f"{self.split}/labels"][real_idx]

        fusion = torch.from_numpy(fusion_np)
        target = torch.from_numpy(target_np)
        name = self.files[index]

        fusion = (fusion - BAND_MEAN) / BAND_STD

        if self.split == "train":
            aug_type = random.choice(list(self.aug_types))
        else:
            aug_type = "orig"

        if aug_type == "hflip":
            fusion = TF.hflip(fusion)
        elif aug_type == "vflip":
            fusion = TF.vflip(fusion)
        elif aug_type == "rot180":
            fusion = torch.rot90(fusion, k=2, dims=[1, 2])

        if self.nodes_backend == "npy":
            base_name = os.path.splitext(name)[0]
            node_name = f"{base_name}_{aug_type}.npy"
            node_path = os.path.join(self.nodes_dir, node_name)

            if not os.path.exists(node_path):
                raise FileNotFoundError(
                    f"Missing precomputed node file: {node_path}. "
                    "Please run precompute_ben_slico_nodes.py first."
                )

            cache_key = ("npy", node_name)
            nodes_np = self._cache_get(cache_key) if self._is_nodes_cache_enabled() else None
            if nodes_np is None:
                nodes_np = np.load(node_path).astype(np.float32, copy=False)
                self._cache_put(cache_key, nodes_np)

            nodes = torch.from_numpy(nodes_np).float()
        else:
            if self.nodes_h5_file is None:
                # Per-worker lazy open; also cache the small index table in memory.
                self.nodes_h5_file = h5py.File(self.nodes_h5_path, "r")
                split_index_key = f"{self.split}/index"
                split_data_key = f"{self.split}/data"
                if split_index_key not in self.nodes_h5_file or split_data_key not in self.nodes_h5_file:
                    raise KeyError(
                        f"Missing split datasets in nodes_h5: {self.split}. "
                        f"Expected keys: {split_index_key}, {split_data_key}"
                    )
                self.nodes_index = self.nodes_h5_file[split_index_key][:]
                self.nodes_data = self.nodes_h5_file[split_data_key]

            aug_idx = self.aug_to_idx[aug_type]
            offset = int(self.nodes_index[real_idx, aug_idx, 0])
            length = int(self.nodes_index[real_idx, aug_idx, 1])

            if length <= 0:
                raise ValueError(
                    f"Invalid nodes length in packed h5. split={self.split}, idx={real_idx}, "
                    f"aug={aug_type}, offset={offset}, length={length}"
                )

            cache_key = ("h5", real_idx, aug_idx)
            nodes_np = self._cache_get(cache_key) if self._is_nodes_cache_enabled() else None
            if nodes_np is None:
                nodes_np = self.nodes_data[offset: offset + length].astype(np.float32, copy=False)
                self._cache_put(cache_key, nodes_np)

            nodes = torch.from_numpy(nodes_np).float()

        return (fusion, name, [self.inp], nodes), target
