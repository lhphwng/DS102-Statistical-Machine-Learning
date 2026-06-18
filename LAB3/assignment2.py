import numpy as np
from sklearn.svm           import LinearSVC, SVC
from sklearn.decomposition import PCA

from datasplit import load_all, normalize
from metrics   import (accuracy, precision, recall, f1_score,
                       print_confusion_matrix, classification_report)

DATA_DIR    = './chest_xray'

PCA_DIM     = 150         # so chieu PCA

# Chon mo hinh: 'LinearSVC' hoac 'SVC'
MODEL       = 'LinearSVC'
C           = 1.0
KERNEL      = 'rbf'       # chi dung khi MODEL = 'SVC': 'linear','rbf','poly'
MAX_ITER    = 5000
RANDOM_SEED = 42


# ══════════════════════════════════════════════
# Pipeline chinh
# ══════════════════════════════════════════════

def main():
    print("=" * 56)
    print("  ASSIGNMENT 2 -- SVM voi scikit-learn")
    print("=" * 56)

    # 1. Tai du lieu
    print("\n[1/4] Tai du lieu...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_all(DATA_DIR)

    # 2. Chuan hoa Z-score
    print("\n[2/4] Chuan hoa Z-score...")
    X_train, X_val, X_test = normalize(X_train, X_val, X_test)

    # 3. Giam chieu PCA (sklearn)
    print("\n[3/4] Giam chieu PCA -> " + str(PCA_DIM) + " chieu...")
    pca = PCA(n_components=PCA_DIM, random_state=RANDOM_SEED)
    X_train = pca.fit_transform(X_train)
    X_val   = pca.transform(X_val)
    X_test  = pca.transform(X_test)
    ratio = round(float(pca.explained_variance_ratio_.sum()) * 100, 1)
    print("  " + str(PCA_DIM) + " thanh phan giai thich " + str(ratio) + "% phuong sai")
    print("  Train: " + str(X_train.shape)
          + "  Val: " + str(X_val.shape)
          + "  Test: " + str(X_test.shape))

    # 4. Huan luyen mo hinh sklearn
    if MODEL == 'LinearSVC':
        label = 'LinearSVC'
        model = LinearSVC(C=C, max_iter=MAX_ITER, dual='auto',
                          random_state=RANDOM_SEED)
    else:
        label = 'SVC (kernel=' + KERNEL + ')'
        model = SVC(C=C, kernel=KERNEL, max_iter=MAX_ITER,
                    random_state=RANDOM_SEED)

    print("\n[4/4] Huan luyen " + label + "  C=" + str(C) + "...")
    model.fit(X_train, y_train)
    print("  Huan luyen hoan tat.")

    # 5. Danh gia
    print("\n" + "=" * 56)
    print("  KET QUA DANH GIA -- " + label)
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

    print("\nAssignment 2 hoan tat.")


if __name__ == '__main__':
    main()
