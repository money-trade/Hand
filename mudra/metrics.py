"""
metrics.py — ตัวชี้วัดการประเมินโมเดล เขียนเองทั้งหมด
accuracy, confusion matrix, precision / recall / F1 (รายคลาส + macro average)
"""
import numpy as np


def accuracy(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean()) if len(y_true) else 0.0


def confusion_matrix(y_true, y_pred, n_classes):
    """แถว = คลาสจริง, คอลัมน์ = คลาสที่ทำนาย"""
    cm = np.zeros((n_classes, n_classes), np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def precision_recall_f1(cm):
    tp = np.diag(cm).astype(np.float64)
    pred_pos = cm.sum(axis=0)
    real_pos = cm.sum(axis=1)
    precision = np.divide(tp, pred_pos, out=np.zeros_like(tp), where=pred_pos > 0)
    recall = np.divide(tp, real_pos, out=np.zeros_like(tp), where=real_pos > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return precision, recall, f1, real_pos


def evaluate(y_true, y_pred, n_classes):
    cm = confusion_matrix(y_true, y_pred, n_classes)
    p, r, f, support = precision_recall_f1(cm)
    present = support > 0
    return {
        "accuracy": accuracy(y_true, y_pred),
        "macro_precision": float(p[present].mean()) if present.any() else 0.0,
        "macro_recall": float(r[present].mean()) if present.any() else 0.0,
        "macro_f1": float(f[present].mean()) if present.any() else 0.0,
        "per_class": {"precision": p.tolist(), "recall": r.tolist(), "f1": f.tolist(),
                      "support": support.tolist()},
        "confusion_matrix": cm.tolist(),
    }


def format_report(result, class_names):
    """ตาราง precision/recall/F1 แบบ markdown"""
    pc = result["per_class"]
    lines = ["| คลาส | Precision | Recall | F1 | จำนวน |", "|---|---:|---:|---:|---:|"]
    for i, c in enumerate(class_names):
        lines.append(f"| {c} | {pc['precision'][i]:.3f} | {pc['recall'][i]:.3f} | "
                     f"{pc['f1'][i]:.3f} | {pc['support'][i]} |")
    lines.append(f"| **macro avg** | {result['macro_precision']:.3f} | {result['macro_recall']:.3f} | "
                 f"{result['macro_f1']:.3f} | {sum(pc['support'])} |")
    lines.append(f"\n**Accuracy = {result['accuracy'] * 100:.2f}%**")
    return "\n".join(lines)


def format_confusion(cm, class_names):
    short = [c[:8] for c in class_names]
    lines = ["| จริง \\ ทำนาย | " + " | ".join(short) + " |",
             "|---|" + "---:|" * len(short)]
    for i, c in enumerate(class_names):
        lines.append(f"| {c} | " + " | ".join(str(v) for v in cm[i]) + " |")
    return "\n".join(lines)
