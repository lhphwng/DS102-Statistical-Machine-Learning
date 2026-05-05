import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt
from Standardization import StandardScaler
from SVM_model import SoftMarginSVM
from Evaluate import precision, recall, f1_score


BASE_DIR = "./chest_xray"

def collect_data(split: str = "train"):
    normal = "NORMAL"
    pneumonia = "PNEUMONIA"

    images = []
    labels = []

    # NORMAL
    for img_file in os.listdir(os.path.join(BASE_DIR, split, normal)):
        path = os.path.join(BASE_DIR, split, normal, img_file)

        img = cv.imread(path)
        if img is None:
            continue

        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.resize(img, (128, 128), interpolation=cv.INTER_LINEAR)
        img = img.reshape(-1) / 255.0

        images.append(img)
        labels.append(-1)   # NORMAL = -1

    # PNEUMONIA
    for img_file in os.listdir(os.path.join(BASE_DIR, split, pneumonia)):
        path = os.path.join(BASE_DIR, split, pneumonia, img_file)

        img = cv.imread(path)
        if img is None:
            continue

        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.resize(img, (128, 128), interpolation=cv.INTER_LINEAR)
        img = img.reshape(-1) / 255.0

        images.append(img)
        labels.append(1)   # PNEUMONIA = 1

    X = np.array(images)
    y = np.array(labels)

    return X, y


# Load data
X_train, y_train = collect_data("train")
X_test, y_test = collect_data("test")

print("Train:", X_train.shape, y_train.shape)
print("Test:", X_test.shape, y_test.shape)

def main(X_train, y_train, X_test, y_test):
    C = 1.0
    LR = 0.0001
    EPOCHS = 10
    BATCH_SIZE = 32

    # ── Standardise ──────────────────────────
    print("\nStandardising …")
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── Train ─────────────────────────────────
    svm = SoftMarginSVM(
        C=C, lr=LR, epochs=EPOCHS,
        batch_size=BATCH_SIZE, lr_schedule="decay"
    )
    svm.fit(X_train, y_train)

    # ── Evaluate ─────────────────────────────
    scaler = StandardScaler()

    # chuẩn hóa
    X_train_std = scaler.fit_transform(X_train)
    X_test_std  = scaler.transform(X_test)

    # train
    svm.fit(X_train_std, y_train)

    # predict
    y_pred = svm.predict(X_test_std)

    # evaluate
    p = precision(y_test, y_pred)
    r = recall(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Precision: {p:.4f}")
    print(f"Recall: {r:.4f}")
    print(f"F1-score: {f1:.4f}")

    # ── Save loss curve ───────────────────────
    plt.figure()
    plt.plot(svm.losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SVM Loss Curve")
    plt.savefig("loss_curve.png")
    print("Saved loss_curve.png")


if __name__ == "__main__":
    main(X_train, y_train, X_test, y_test)