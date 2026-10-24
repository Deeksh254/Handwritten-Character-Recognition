"""Pure-NumPy replacements for sklearn's confusion_matrix / classification_report.

Used so the project does not need scikit-learn (and therefore scipy), whose DLLs are
blocked by some Windows security policies.
"""
import numpy as np


def confusion_matrix(y_true, y_pred, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def per_class_scores(cm: np.ndarray):
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)
    predicted = cm.sum(axis=0).astype(float)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return precision, recall, f1, support


def classification_report(cm: np.ndarray, classes, digits: int = 4) -> str:
    precision, recall, f1, support = per_class_scores(cm)
    total = support.sum()
    acc = np.diag(cm).sum() / total if total else 0.0
    name_w = max(len(str(c)) for c in list(classes) + ["weighted avg"])
    head = f"{'':>{name_w}} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>9}"
    rows = [head, ""]
    for i, c in enumerate(classes):
        rows.append(f"{str(c):>{name_w}} {precision[i]:>10.{digits}f} {recall[i]:>10.{digits}f} "
                    f"{f1[i]:>10.{digits}f} {int(support[i]):>9d}")
    rows.append("")
    rows.append(f"{'accuracy':>{name_w}} {'':>10} {'':>10} {acc:>10.{digits}f} {int(total):>9d}")
    rows.append(f"{'macro avg':>{name_w}} {precision.mean():>10.{digits}f} {recall.mean():>10.{digits}f} "
                f"{f1.mean():>10.{digits}f} {int(total):>9d}")
    w = support / total if total else support
    rows.append(f"{'weighted avg':>{name_w}} {(precision * w).sum():>10.{digits}f} "
                f"{(recall * w).sum():>10.{digits}f} {(f1 * w).sum():>10.{digits}f} {int(total):>9d}")
    return "\n".join(rows)
