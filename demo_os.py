import argparse
import os
import sys
import torch
import torch.optim
import csv
import datetime
import numpy as np
import random

# 保证 src 模块能被正确导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import DSDLMultiLabelMAPEngine
from ben_dataset import BEN10Dataset
from loss import MyLoss
from models import load_model


# ===============================
# 默认路径
# ===============================
DEFAULT_DATA_PATH = "/media/sata/xyx/BigEarthNet/dataset"
DEFAULT_CHECKPOINT_PATH = "/media/sata/xyx/BigEarthNet/checkpoints/ben_sar/"
DEFAULT_LOG_PATH = "/media/sata/xyx/BigEarthNet/logs/ben_sar/"

# 保证实验可复现的随机种子
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(3407)

# =================================
# 日志记录类
# =================================
class TrainingLogger:
    def __init__(self, log_dir=DEFAULT_LOG_PATH):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"training_log_{timestamp}.csv")
        self.init_csv()

    def init_csv(self):
        with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 新增列：Macro_P, Macro_R, Macro_F1
            writer.writerow([
                'timestamp', 'epoch', 'phase', 'loss',
                'backbone_lr', 'semantic_lr',
                'mAP', 'Macro_P', 'Macro_R', 'Macro_F1', 'Micro_F1', 'Threshold',
                'AP_per_class', 'F1_per_class', 'epoch_time'
            ])

    def log_epoch(self, epoch, phase, loss, lr, metrics, epoch_time):
        # ... (lr 处理逻辑不变) ...
        if isinstance(lr, (list, tuple)):
            backbone_lr, semantic_lr = lr[0], lr[-1]
        elif hasattr(lr, "__len__"):
            backbone_lr, semantic_lr = lr[0], lr[-1]
        elif lr is None:
            backbone_lr = semantic_lr = ''
        else:
            backbone_lr = semantic_lr = lr

        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                epoch,
                phase,
                f"{loss:.6f}" if loss is not None else '',
                f"{float(backbone_lr):.8f}" if backbone_lr != '' else '',
                f"{float(semantic_lr):.8f}" if semantic_lr != '' else '',
                metrics.get('mAP', ''),
                metrics.get('Macro_P', ''),
                metrics.get('Macro_R', ''),
                metrics.get('Macro_F1', ''),
                metrics.get('Micro_F1', ''),
                metrics.get('Threshold', ''),
                metrics.get('AP_per_class', ''),
                metrics.get('F1_per_class', ''),
                f"{epoch_time:.3f}"
            ])

    def log_best_model(self, best_metrics):
        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([])
            writer.writerow(["==== Best Model Summary (Based on Micro-F1) ===="])
            writer.writerow([
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "BEST_MODEL",
                "", "", "", "",
                best_metrics.get('mAP', ''),
                best_metrics.get('Macro_P', ''),
                best_metrics.get('Macro_R', ''),
                best_metrics.get('Macro_F1', ''),
                best_metrics.get('Micro_F1', ''),
                best_metrics.get('Threshold', ''),
                best_metrics.get('AP_per_class', ''),
                best_metrics.get('F1_per_class', ''),
                ""
            ])


# ===============================
# 参数解析
# ===============================
def build_parser():
    parser = argparse.ArgumentParser(description='OS Dataset Training (Optical + SAR Fusion)')

    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--data', default=DEFAULT_DATA_PATH, type=str)
    data_group.add_argument('--image-size', '-i', default=128, type=int)
    data_group.add_argument(
        '--h5-path',
        default='',
        type=str,
        help='optional override for BEN h5 file path (supports h5 on another disk)',
    )
    data_group.add_argument(
        '--index-subdir',
        default='',
        type=str,
        help='optional override for processed_pt index subdir, e.g. processed_pt_<image_size>_clean622',
    )
    data_group.add_argument(
        '--embedding-path',
        default='',
        type=str,
        help='optional override for embedding pkl path',
    )
    data_group.add_argument(
        '--train-max-samples',
        default=0,
        type=int,
        help='for quick check: use only first N train samples',
    )
    data_group.add_argument(
        '--val-max-samples',
        default=0,
        type=int,
        help='for quick check: use only first N val/test samples',
    )

    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--device_ids', '--device-ids', default=[0], type=int, nargs='+')
    train_group.add_argument('-j', '--workers', default=4, type=int)
    train_group.add_argument(
        '--prefetch-factor',
        default=2,
        type=int,
        help='DataLoader prefetch_factor when workers > 0',
    )
    train_group.add_argument(
        '--val-persistent-workers',
        action='store_true',
        help='keep val workers persistent across epochs (default: off for memory stability)',
    )
    train_group.add_argument('--epochs', default=100, type=int)
    train_group.add_argument('-b', '--batch-size', default=32, type=int)
    train_group.add_argument('--lr', default=0.001, type=float)
    train_group.add_argument('--lrp', default=0.1, type=float)
    train_group.add_argument('--weight-decay', default=1e-4, type=float)
    train_group.add_argument('--resume', default='', type=str, help='path to checkpoint to resume from')
    train_group.add_argument('--evaluate', action='store_true')
    train_group.add_argument('--log-dir', default=DEFAULT_LOG_PATH, type=str)
    train_group.add_argument(
        '--early-stop',
        action='store_true',
        help='enable early stopping based on validation Micro-F1',
    )
    train_group.add_argument(
        '--patience',
        default=15,
        type=int,
        help='number of epochs with no improvement before stopping',
    )

    loss_group = parser.add_argument_group('Loss')
    loss_group.add_argument('--lambd', default=0.001, type=float)
    loss_group.add_argument('--beta', default=0.005, type=float)

    sar_group = parser.add_argument_group('SAR Backbone')
    sar_group.add_argument('--sar-patch-size', default=8, type=int)
    sar_group.add_argument('--sar-embed-dim', default=64, type=int)
    sar_group.add_argument('--sar-num-vig-blocks', default=2, type=int)
    sar_group.add_argument('--sar-num-segments', default=64, type=int)
    sar_group.add_argument('--sar-num-edges', default=9, type=int)
    sar_group.add_argument('--sar-head-num', default=1, type=int)
    sar_group.add_argument('--sar-drop-path', default=0.05, type=float)

    node_group = parser.add_argument_group('Nodes')
    node_group.add_argument(
        '--nodes-dir-train',
        default='',
        type=str,
        help='optional override for train nodes directory',
    )
    node_group.add_argument(
        '--nodes-dir-val',
        default='',
        type=str,
        help='optional override for val nodes directory',
    )
    node_group.add_argument(
        '--nodes-backend',
        default='auto',
        choices=['auto', 'npy', 'h5'],
        help='node loading backend: auto prefers packed h5 when available',
    )
    node_group.add_argument(
        '--nodes-h5-path',
        default='',
        type=str,
        help='packed nodes h5 path, default: <data>/ben_slico_nodes_seg*_patch*_img<image_size>.h5',
    )

    return parser

# ===============================
# main function
# ===============================
def main_os():
    args = build_parser().parse_args()
    embedding_path = args.embedding_path or os.path.join(
        args.data, "embeddings", "bigearthnet19_glove_word2vec.pkl"
    )
    h5_path = args.h5_path if args.h5_path else None
    index_subdir = args.index_subdir if args.index_subdir else None

    global logger
    logger = TrainingLogger(args.log_dir)

    print("############################################")
    print(" Optical + SAR Fusion DSDL Training ")
    print("############################################")
    print(f"Data path: {args.data}")
    print(f"Embedding path: {embedding_path}")
    if h5_path:
        print(f"Data h5 override: {h5_path}")
    if index_subdir:
        print(f"Index subdir override: {index_subdir}")
    print(f"Log path:  {logger.log_dir}")
    if args.resume:
        print(f"Resume from checkpoint: {args.resume}")

    # ============ Dataset ============
    train_dataset = BEN10Dataset(
        root=args.data,
        split="train",
        transform=None,
        inp_name=embedding_path,
        image_size=args.image_size,
        max_samples=args.train_max_samples,
        num_segments=args.sar_num_segments,
        patch_size=args.sar_patch_size,
        nodes_dir=args.nodes_dir_train if args.nodes_dir_train else None,
        nodes_backend=args.nodes_backend,
        nodes_h5_path=args.nodes_h5_path if args.nodes_h5_path else None,
        h5_path=h5_path,
        index_subdir=index_subdir,
    )

    val_dataset = BEN10Dataset(
        root=args.data,
        split="val",   # 建议这里先用 val，不要先用 test
        transform=None,
        inp_name=embedding_path,
        image_size=args.image_size,
        max_samples=args.val_max_samples,
        num_segments=args.sar_num_segments,
        patch_size=args.sar_patch_size,
        nodes_dir=args.nodes_dir_val if args.nodes_dir_val else None,
        nodes_backend=args.nodes_backend,
        nodes_h5_path=args.nodes_h5_path if args.nodes_h5_path else None,
        h5_path=h5_path,
        index_subdir=index_subdir,
    )

    # ============ Model ============
    num_classes = 19
    model = load_model(
        num_classes=num_classes,
        alpha=args.lambd,
        sar_patch_size=args.sar_patch_size,
        sar_embed_dim=args.sar_embed_dim,
        sar_num_vig_blocks=args.sar_num_vig_blocks,
        sar_num_segments=args.sar_num_segments,
        sar_num_edges=args.sar_num_edges,
        sar_head_num=args.sar_head_num,
        sar_drop_path=args.sar_drop_path,
    )

    # ============ Loss & Optimizer ============
    criterion = MyLoss(args.lambd, args.beta)
    #  AdamW 优化器
    optimizer = torch.optim.AdamW(
        model.get_config_optim(args.lr, args.lrp),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # ============ Engine ============
    state = {
        'batch_size': args.batch_size,
        'image_size': args.image_size,
        'max_epochs': args.epochs,
        'evaluate': args.evaluate,
        'resume': args.resume,
        'num_classes': num_classes,
        'workers': args.workers,
        'prefetch_factor': args.prefetch_factor,
        'val_persistent_workers': args.val_persistent_workers,
        'lr': args.lr,
        'device_ids': args.device_ids,
        'dataset': 'os',
        'logger': logger,
        'save_model_path': DEFAULT_CHECKPOINT_PATH,
        'early_stop': args.early_stop,
        'patience': args.patience,
    }

    engine = DSDLMultiLabelMAPEngine(state)
    best_score = engine.learning(model, criterion, train_dataset, val_dataset, optimizer)

    # 打印和记录最佳结果
    best_metrics = getattr(engine, 'best_metrics', {})
    if not best_metrics:
        best_metrics = {'Micro_F1': f"{best_score:.4f}"}

    logger.log_best_model(best_metrics)

    print("############################################")
    print(" Training Complete! ")
    print("############################################")
    print(f"Best Micro-F1 = {best_metrics.get('Micro_F1', best_score)}")
    print(f"Best Epoch    = {engine.state.get('best_epoch', 'N/A')}")
    print(f"Macro-F1      = {best_metrics.get('Macro_F1', 'N/A')}")
    print(f"Macro-P       = {best_metrics.get('Macro_P', 'N/A')}")
    print(f"Macro-R       = {best_metrics.get('Macro_R', 'N/A')}")
    print(f"Threshold     = {best_metrics.get('Threshold', 'N/A')}")
    print("--------------------------------------------")
    print(f"Best model saved in: {DEFAULT_CHECKPOINT_PATH}")
    print(f"Log file saved in:   {logger.log_file}")
    print("############################################")

if __name__ == "__main__":
    main_os()
