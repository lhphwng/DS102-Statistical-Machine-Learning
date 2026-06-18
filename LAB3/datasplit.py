import numpy as np
from dataloader import load_split


def load_all(data_dir):
    """
    Tải cả 3 split từ thư mục dataset có sẵn.

    Tham số:
        data_dir : thư mục gốc chứa train/ val/ test/

    Trả về:
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test
    """
    print("=== Dang tai du lieu tu: " + data_dir + " ===")

    X_train, y_train = load_split(data_dir, 'train')
    X_val,   y_val   = load_split(data_dir, 'val')
    X_test,  y_test  = load_split(data_dir, 'test')

    print("")
    print("Ket qua:")
    print("  Train : " + str(len(y_train)) + " mau"
          + "  (NORMAL=" + str(int(np.sum(y_train == 0)))
          + ", PNEUMONIA=" + str(int(np.sum(y_train == 1))) + ")")
    print("  Val   : " + str(len(y_val)) + " mau"
          + "  (NORMAL=" + str(int(np.sum(y_val == 0)))
          + ", PNEUMONIA=" + str(int(np.sum(y_val == 1))) + ")")
    print("  Test  : " + str(len(y_test)) + " mau"
          + "  (NORMAL=" + str(int(np.sum(y_test == 0)))
          + ", PNEUMONIA=" + str(int(np.sum(y_test == 1))) + ")")

    return X_train, y_train, X_val, y_val, X_test, y_test


def normalize(X_train, X_val, X_test):
    """
    Chuẩn hoá Z-score.
    Tính mean và std CHI tren X_train, ap dung cho ca 3 tap
    de tranh data leakage.

    Tham số:
        X_train, X_val, X_test : numpy array (N, D)

    Trả về:
        X_train, X_val, X_test da chuan hoa
    """
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0) + 1e-8   # tranh chia 0

    X_train_n = (X_train - mean) / std
    X_val_n   = (X_val   - mean) / std
    X_test_n  = (X_test  - mean) / std

    return X_train_n, X_val_n, X_test_n
