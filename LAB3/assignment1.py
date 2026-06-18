import numpy as np

from datasplit import load_all, normalize
from svm       import SVM
from metrics   import (accuracy, precision, recall, f1_score,
                       print_confusion_matrix, classification_report)


DATA_DIR      = './chest_xray'

PCA_DIM       = 150       # so chieu sau PCA (giam tu 16384)

C             = 1.0       # tham so regularization SVM
LEARNING_RATE = 0.01      # learning rate SGD ban dau
N_EPOCHS      = 100       # so vong lap
BATCH_SIZE    = 64        # kich thuoc mini-batch
RANDOM_SEED   = 42


# ══════════════════════════════════════════════
# PCA bang NumPy (SVD)
# ══════════════════════════════════════════════

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean_        = None   # shape (D,)
        self.components_  = None   # shape (n_components, D)

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        print("  Dang tinh SVD " + str(Xc.shape) + "...")
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.components_ = Vt[:self.n_components]
        ratio = round(float((S**2)[:self.n_components].sum() / (S**2).sum()) * 100, 1)
        print("  " + str(self.n_components)
              + " thanh phan giai thich " + str(ratio) + "% phuong sai")
        return self

    def transform(self, X):
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ══════════════════════════════════════════════
# Pipeline chinh
# ══════════════════════════════════════════════

def main():
    print("=" * 56)
    print("  ASSIGNMENT 1 -- Soft-Margin SVM (NumPy tu dau)")
    print("=" * 56)

    # 1. Tai du lieu
    print("\n[1/4] Tai du lieu...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_all(DATA_DIR)

    # 2. Chuan hoa Z-score
    print("\n[2/4] Chuan hoa Z-score...")
    X_train, X_val, X_test = normalize(X_train, X_val, X_test)

    # 3. Giam chieu PCA
    print("\n[3/4] Giam chieu PCA -> " + str(PCA_DIM) + " chieu...")
    pca = PCA(PCA_DIM)
    X_train = pca.fit_transform(X_train)
    X_val   = pca.transform(X_val)
    X_test  = pca.transform(X_test)
    print("  Train: " + str(X_train.shape)
          + "  Val: " + str(X_val.shape)
          + "  Test: " + str(X_test.shape))

    # 4. Huan luyen SVM
    print("\n[4/4] Huan luyen SVM (NumPy)...")
    model = SVM(
        C=C,
        learning_rate=LEARNING_RATE,
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        random_seed=RANDOM_SEED,
    )
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val, verbose=True)

    # 5. Danh gia
    print("\n" + "=" * 56)
    print("  KET QUA DANH GIA")
    print("=" * 56)

    for name, Xs, ys in [("TRAIN",      X_train, y_train),
                          ("VALIDATION", X_val,   y_val),
                          ("TEST",       X_test,  y_test)]:
        yp = model.predict(Xs)
        print("\n-- " + name + " --")
        print("  Accuracy  : " + str(round(accuracy (ys, yp), 4)))
        print("  Precision : " + str(round(precision(ys, yp), 4)))
        print("  Recall    : " + str(round(recall   (ys, yp), 4)))
        print("  F1-Score  : " + str(round(f1_score (ys, yp), 4)))
        print_confusion_matrix(ys, yp)

    print("\n-- BAO CAO CHI TIET -- TAP TEST --")
    classification_report(y_test, model.predict(X_test))

    print("\nAssignment 1 hoan tat.")


if __name__ == '__main__':
    main()
