"""Metrics computation for SENTINEL evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class DetectionMetrics:
    """Standard detection metrics."""

    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        """Precision = TP / (TP + FP)"""
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        """Recall = TP / (TP + FN)"""
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)

    @property
    def f1_score(self) -> float:
        """F1 = 2 * (precision * recall) / (precision + recall)"""
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)

    @property
    def accuracy(self) -> float:
        """Accuracy = (TP + TN) / (TP + TN + FP + FN)"""
        total = (
            self.true_positives
            + self.true_negatives
            + self.false_positives
            + self.false_negatives
        )
        if total == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / total

    @property
    def false_positive_rate(self) -> float:
        """FPR = FP / (FP + TN)"""
        if self.false_positives + self.true_negatives == 0:
            return 0.0
        return self.false_positives / (self.false_positives + self.true_negatives)

    @property
    def false_negative_rate(self) -> float:
        """FNR = FN / (FN + TP)"""
        if self.false_negatives + self.true_positives == 0:
            return 0.0
        return self.false_negatives / (self.false_negatives + self.true_positives)


def compute_metrics(
    predictions: Sequence[bool],
    labels: Sequence[bool],
) -> DetectionMetrics:
    """
    Compute detection metrics from predictions and labels.

    Args:
        predictions: Model predictions (True = injection detected)
        labels: Ground truth labels (True = actual injection)

    Returns:
        DetectionMetrics with TP, FP, TN, FN counts
    """
    tp = fp = tn = fn = 0

    for pred, label in zip(predictions, labels):
        if pred and label:
            tp += 1
        elif pred and not label:
            fp += 1
        elif not pred and not label:
            tn += 1
        else:  # not pred and label
            fn += 1

    return DetectionMetrics(
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
    )


def compute_auc(
    confidences: Sequence[float],
    labels: Sequence[bool],
    num_thresholds: int = 100,
) -> float:
    """
    Compute Area Under the ROC Curve.

    Args:
        confidences: Model confidence scores
        labels: Ground truth labels
        num_thresholds: Number of threshold points

    Returns:
        AUC score between 0 and 1
    """
    if not confidences or not labels:
        return 0.0

    thresholds = [i / num_thresholds for i in range(num_thresholds + 1)]
    tpr_list = []
    fpr_list = []

    for threshold in thresholds:
        predictions = [c >= threshold for c in confidences]
        metrics = compute_metrics(predictions, labels)
        tpr_list.append(metrics.recall)  # TPR = recall
        fpr_list.append(metrics.false_positive_rate)

    # Compute AUC using trapezoidal rule
    auc = 0.0
    for i in range(1, len(fpr_list)):
        # Note: FPR decreases as threshold increases
        width = fpr_list[i - 1] - fpr_list[i]
        height = (tpr_list[i - 1] + tpr_list[i]) / 2
        auc += width * height

    return auc


def compute_latency_percentiles(
    latencies: Sequence[float],
    percentiles: Sequence[float] = (0.5, 0.9, 0.95, 0.99),
) -> dict[str, float]:
    """
    Compute latency percentiles.

    Args:
        latencies: List of latency measurements in ms
        percentiles: Percentiles to compute

    Returns:
        Dict mapping percentile names to values
    """
    if not latencies:
        return {f"p{int(p*100)}": 0.0 for p in percentiles}

    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)

    result = {}
    for p in percentiles:
        idx = int(p * n)
        idx = min(idx, n - 1)
        result[f"p{int(p*100)}"] = sorted_latencies[idx]

    result["mean"] = sum(latencies) / n
    result["min"] = sorted_latencies[0]
    result["max"] = sorted_latencies[-1]

    return result
