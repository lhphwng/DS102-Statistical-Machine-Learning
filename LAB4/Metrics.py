import numpy as np

def f1_score(y_true, y_pred):

    tp = np.sum(
        (y_true == 1)
        & (y_pred == 1)
    )

    fp = np.sum(
        (y_true == 0)
        & (y_pred == 1)
    )

    fn = np.sum(
        (y_true == 1)
        & (y_pred == 0)
    )

    precision = tp / (
        tp + fp + 1e-10
    )

    recall = tp / (
        tp + fn + 1e-10
    )

    f1 = (
        2
        * precision
        * recall
        / (
            precision
            + recall
            + 1e-10
        )
    )

    return f1