import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

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

def train_sklearn_svm(X_train, y_train, X_test, y_test):
    print("\n[Sklearn SVM] Standardising...")
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print("[Sklearn SVM] Training...")
    model = LinearSVC(C=1.0, max_iter=1000)
    model.fit(X_train, y_train)

    print("[Sklearn SVM] Evaluating...")
    y_pred = model.predict(X_test)

    p = precision(y_test, y_pred)
    r = recall(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"[Sklearn] Precision: {p:.4f}")
    print(f"[Sklearn] Recall:    {r:.4f}")
    print(f"[Sklearn] F1-score:  {f1:.4f}")

    return p, r, f1

if __name__ == "__main__":
    train_sklearn_svm(X_train, y_train, X_test, y_test)