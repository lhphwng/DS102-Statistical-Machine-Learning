import os
import numpy as np
import cv2

IMAGE_SIZE      = (128, 128)
LABEL_NORMAL    = 0
LABEL_PNEUMONIA = 1


def load_images_from_folder(folder_path, label):
    X, y = [], []

    for filename in sorted(os.listdir(folder_path)):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(folder_path, filename)

        # Đọc ảnh dưới dạng grayscale thẳng luôn
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("  [LỖI] Không đọc được: " + filename)
            continue

        # Resize về 128x128
        img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

        # Normalize [0, 1] và flatten thành vector 16384 chiều
        img = img.astype(np.float32) / 255.0
        X.append(img.flatten())
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def load_split(data_dir, split):
    """
    Tải một split (train / val / test).

    Tham số:
        data_dir : thư mục gốc, ví dụ './chest_xray'
        split    : 'train', 'val', hoặc 'test'

    Trả về:
        X : numpy array (N, 16384), float32
        y : numpy array (N,),       int32
    """
    all_X, all_y = [], []

    for class_name, label in [('NORMAL', LABEL_NORMAL), ('PNEUMONIA', LABEL_PNEUMONIA)]:
        folder = os.path.join(data_dir, split, class_name)

        if not os.path.isdir(folder):
            raise FileNotFoundError("Không tìm thấy: " + folder)

        X_cls, y_cls = load_images_from_folder(folder, label)
        print("  " + split + "/" + class_name
              + " -> " + str(len(y_cls)) + " anh (nhan=" + str(label) + ")")

        all_X.append(X_cls)
        all_y.append(y_cls)

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    return X, y
