import os
import torch
import h5py
from tqdm import tqdm

# 请确保这里的 ROOT 是你存放数据集的目录
ROOT = "/media/sata/xyx/BigEarthNet/dataset"
DATA_DIR = os.path.join(ROOT, "processed_pt_256_clean622") # 指向你的 256 尺寸 pt 文件夹
H5_PATH = os.path.join(ROOT, "ben_10p_clean_622_256.h5")    # 即将生成的终极 h5 文件

def create_h5():
    # 'w' 模式创建并打开 h5 文件
    with h5py.File(H5_PATH, 'w') as h5f:
        # for split in ["train", "val", "test"]:
        for split in ["train", "val"]:
            txt_path = os.path.join(DATA_DIR, f"{split}.txt")
            if not os.path.exists(txt_path): 
                continue

            with open(txt_path, 'r', encoding='utf-8') as f:
                files = [line.strip() for line in f if line.strip()]

            N = len(files)
            print(f"开始打包 {split} 集 ({N} 个样本)...")

            # 在 h5 中创建数据集，指定 chunks 优化随机读取性能
            img_ds = h5f.create_dataset(f"{split}/images", shape=(N, 5, 256, 256), dtype='float32', chunks=(1, 5, 256, 256))
            lbl_ds = h5f.create_dataset(f"{split}/labels", shape=(N, 19), dtype='float32')

            # 遍历写入
            for i, fname in enumerate(tqdm(files)):
                pt_path = os.path.join(DATA_DIR, split, fname)
                sample = torch.load(pt_path, map_location="cpu", weights_only=True)
                
                img_ds[i] = sample["image"].numpy()
                lbl_ds[i] = sample["label"].numpy()

if __name__ == "__main__":
    create_h5()
