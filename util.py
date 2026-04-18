import torch
import numpy as np


class AveragePrecisionMeter(object):
    """
    计算多标签任务指标：
    - Macro Precision / Recall / F1
    - Micro F1
    - mAP
    使用固定阈值进行评估（默认 0.5）。
    """

    def __init__(self, difficult_examples=False):
        self.difficult_examples = difficult_examples
        self.reset()

    def reset(self):
        self.scores = torch.FloatTensor()
        self.targets = torch.LongTensor()

    def add(self, output, target):
        """
        output: [N, C] logits
        target: [N, C] {0,1}
        """
        if not torch.is_tensor(output):
            output = torch.tensor(output)
        if not torch.is_tensor(target):
            target = torch.tensor(target)

        output = output.detach().cpu()
        target = target.detach().cpu()

        if self.scores.numel() == 0:
            self.scores = output
            self.targets = target
        else:
            self.scores = torch.cat([self.scores, output], dim=0)
            self.targets = torch.cat([self.targets, target], dim=0)

    def value(self):
        """返回每个类别的 AP"""
        if self.scores.numel() == 0:
            return torch.zeros(1)
        ap = torch.zeros(self.scores.size(1))
        for k in range(self.scores.size(1)):
            ap[k] = self.average_precision(self.scores[:, k], self.targets[:, k])
        return ap

    @staticmethod
    def average_precision(output, target):
        """单类 AP 计算"""
        sorted_scores, indices = torch.sort(output, descending=True)
        target_sorted = target[indices]
        tp_cumsum = torch.cumsum(target_sorted, dim=0)
        total_pos = tp_cumsum[-1].item()

        if total_pos == 0:
            return 0.0

        k_indices = torch.arange(1, len(target) + 1, dtype=torch.float32)
        precision_at_k = tp_cumsum / k_indices
        ap = (precision_at_k * target_sorted).sum() / total_pos
        return ap.item()

    def _compute_metrics_from_probs(self, probs, targets, threshold, ap_per_class):
        preds = (probs >= threshold).astype(np.float32)

        tp = np.sum((preds == 1) & (targets == 1), axis=0)
        fp = np.sum((preds == 1) & (targets == 0), axis=0)
        fn = np.sum((preds == 0) & (targets == 1), axis=0)

        p_class = tp / (tp + fp + 1e-10)
        r_class = tp / (tp + fn + 1e-10)
        f1_class = 2 * p_class * r_class / (p_class + r_class + 1e-10)

        macro_p = np.mean(p_class) * 100.0
        macro_r = np.mean(r_class) * 100.0
        macro_f1 = 2 * macro_p * macro_r / (macro_p + macro_r + 1e-10)

        tp_micro = np.sum(tp)
        fp_micro = np.sum(fp)
        fn_micro = np.sum(fn)

        micro_p = tp_micro / (tp_micro + fp_micro + 1e-10) * 100.0
        micro_r = tp_micro / (tp_micro + fn_micro + 1e-10) * 100.0
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + 1e-10)

        mAP = np.mean(ap_per_class)

        return {
            "mAP": mAP,
            "Macro_P": macro_p,
            "Macro_R": macro_r,
            "Macro_F1": macro_f1,
            "Micro_F1": micro_f1,
            "Per_Class_AP": ap_per_class,
            "Per_Class_F1": f1_class * 100.0,
            "Threshold": threshold,
        }

    def compute_paper_metrics(self, threshold=0.5):
        """
        用指定阈值计算指标。
        """
        if self.scores.numel() == 0:
            return {
                "mAP": 0.0,
                "Macro_P": 0.0,
                "Macro_R": 0.0,
                "Macro_F1": 0.0,
                "Micro_F1": 0.0,
                "Per_Class_AP": np.zeros((1,), dtype=np.float32),
                "Per_Class_F1": np.zeros((1,), dtype=np.float32),
                "Threshold": float(threshold),
            }

        probs = torch.sigmoid(self.scores).numpy()
        targets = self.targets.numpy()
        ap_per_class = self.value().numpy()
        return self._compute_metrics_from_probs(
            probs=probs,
            targets=targets,
            threshold=float(threshold),
            ap_per_class=ap_per_class,
        )

    def find_best_threshold(self, thresholds=None, metric_key="Micro_F1"):
        """
        在验证集上搜索最优阈值，默认最大化 Micro-F1。
        """
        if self.scores.numel() == 0:
            default_threshold = 0.5
            return default_threshold, self.compute_paper_metrics(threshold=default_threshold)

        if thresholds is None:
            thresholds = np.linspace(0.0, 1.0, 101, dtype=np.float32)

        thresholds = np.asarray(list(thresholds), dtype=np.float32)
        if thresholds.size == 0:
            raise ValueError("thresholds must not be empty")

        probs = torch.sigmoid(self.scores).numpy()
        targets = self.targets.numpy()
        ap_per_class = self.value().numpy()

        first_th = float(thresholds[0])
        best_threshold = first_th
        best_metrics = self._compute_metrics_from_probs(
            probs=probs,
            targets=targets,
            threshold=first_th,
            ap_per_class=ap_per_class,
        )
        best_score = float(best_metrics.get(metric_key, float("-inf")))

        for th in thresholds[1:]:
            th = float(th)
            metrics = self._compute_metrics_from_probs(
                probs=probs,
                targets=targets,
                threshold=th,
                ap_per_class=ap_per_class,
            )
            score = float(metrics.get(metric_key, float("-inf")))

            # Tie-breaker: same score时优先更接近0.5的阈值，避免极端阈值抖动。
            if (
                score > best_score + 1e-12
                or (
                    abs(score - best_score) <= 1e-12
                    and abs(th - 0.5) < abs(best_threshold - 0.5)
                )
            ):
                best_score = score
                best_threshold = th
                best_metrics = metrics

        return best_threshold, best_metrics
