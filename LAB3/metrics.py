import numpy as np


def _count(y_true, y_pred, positive=1):
    """Dem TP, FP, FN, TN."""
    neg = 1 - positive
    TP = int(np.sum((y_true == positive) & (y_pred == positive)))
    FP = int(np.sum((y_true == neg)      & (y_pred == positive)))
    FN = int(np.sum((y_true == positive) & (y_pred == neg)))
    TN = int(np.sum((y_true == neg)      & (y_pred == neg)))
    return TP, FP, FN, TN


def precision(y_true, y_pred, positive=1):
    """
    Precision = TP / (TP + FP)

    Trong so cac mau duoc du doan la DUONG,
    bao nhieu phan tram thuc su la DUONG?
    """
    TP, FP, FN, TN = _count(y_true, y_pred, positive)
    return TP / (TP + FP) if (TP + FP) > 0 else 0.0


def recall(y_true, y_pred, positive=1):
    """
    Recall = TP / (TP + FN)

    Trong so cac mau THUC SU la DUONG,
    mo hinh phat hien dung bao nhieu phan tram?
    """
    TP, FP, FN, TN = _count(y_true, y_pred, positive)
    return TP / (TP + FN) if (TP + FN) > 0 else 0.0


def f1_score(y_true, y_pred, positive=1):
    """
    F1 = 2 * P * R / (P + R)

    Trung binh dieu hoa cua Precision va Recall.
    """
    p = precision(y_true, y_pred, positive)
    r = recall   (y_true, y_pred, positive)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def accuracy(y_true, y_pred):
    """Accuracy = so du doan dung / tong so mau."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def print_confusion_matrix(y_true, y_pred):
    """
    In ma tran nham lan dang:

        Confusion Matrix:
                         Pred NORMAL   Pred PNEUMONIA
        True NORMAL          TN             FP
        True PNEUMONIA       FN             TP
    """
    TP, FP, FN, TN = _count(y_true, y_pred, positive=1)
    print("  Confusion Matrix:")
    print("  " + " " * 22 + "Pred NORMAL   Pred PNEUMONIA")
    print("  " + "True NORMAL".ljust(22) + str(TN).center(13) + str(FP).center(15))
    print("  " + "True PNEUMONIA".ljust(22) + str(FN).center(13) + str(TP).center(15))


def classification_report(y_true, y_pred):
    """
    In bao cao day du va tra ve dict chua cac chi so.

    Chi so cho tung class va macro average:
        Precision, Recall, F1-Score, Support
    """
    p1 = precision(y_true, y_pred, positive=1)
    r1 = recall   (y_true, y_pred, positive=1)
    f1 = f1_score (y_true, y_pred, positive=1)

    p0 = precision(y_true, y_pred, positive=0)
    r0 = recall   (y_true, y_pred, positive=0)
    f0 = f1_score (y_true, y_pred, positive=0)

    acc     = accuracy(y_true, y_pred)
    macro_p = (p0 + p1) / 2
    macro_r = (r0 + r1) / 2
    macro_f = (f0 + f1) / 2

    n0 = int(np.sum(np.asarray(y_true) == 0))
    n1 = int(np.sum(np.asarray(y_true) == 1))
    N  = n0 + n1

    w = 14
    print("  " + "-" * 62)
    print("  " + "Class".ljust(18)
          + "Precision".rjust(w) + "Recall".rjust(w)
          + "F1-Score".rjust(w) + "Support".rjust(w))
    print("  " + "-" * 62)
    print("  " + "NORMAL (0)".ljust(18)
          + str(round(p0, 4)).rjust(w) + str(round(r0, 4)).rjust(w)
          + str(round(f0, 4)).rjust(w) + str(n0).rjust(w))
    print("  " + "PNEUMONIA (1)".ljust(18)
          + str(round(p1, 4)).rjust(w) + str(round(r1, 4)).rjust(w)
          + str(round(f1, 4)).rjust(w) + str(n1).rjust(w))
    print("  " + "-" * 62)
    print("  " + "Macro Avg".ljust(18)
          + str(round(macro_p, 4)).rjust(w) + str(round(macro_r, 4)).rjust(w)
          + str(round(macro_f, 4)).rjust(w) + str(N).rjust(w))
    print("  " + "Accuracy".ljust(18)
          + " " * (w * 2) + str(round(acc, 4)).rjust(w) + str(N).rjust(w))
    print("  " + "-" * 62)

    return {
        'accuracy' : acc,
        'precision': p1,
        'recall'   : r1,
        'f1_score' : f1,
        'macro_p'  : macro_p,
        'macro_r'  : macro_r,
        'macro_f1' : macro_f,
    }
