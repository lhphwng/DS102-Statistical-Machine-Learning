import numpy as np


def generate_gaussian_cluster(mean, cov, n_samples, rng):
    mean = np.array(mean, dtype=float)
    cov  = np.array(cov,  dtype=float)
    D = mean.shape[0]

    # Phân rã Cholesky: sigma = L @ L.T
    L = np.linalg.cholesky(cov)

    # Sinh Z ~ N(0, I), shape (D, n_samples)
    Z = rng.standard_normal((D, n_samples))

    # Biến đổi tuyến tính: X = mu + L @ Z, transpose về (n_samples, D)
    X = (L @ Z).T + mean
    return X


# ---------------------------------------------------------------------------
# Các hàm sinh dữ liệu cho từng bài tập
# ---------------------------------------------------------------------------

def generate_data_a1(seed=42):
    """
    Bài A1: 600 điểm, 3 cụm đều (200 điểm/cụm).
    Tâm: (2,2), (8,3), (3,6). Cùng ma trận Σ = [[1,0],[0,1]].

    Returns:
        X (np.ndarray): shape (600, 2)
        y (np.ndarray): Nhãn cụm thực (0, 1, 2), shape (600,)
    """
    rng = np.random.default_rng(seed)

    means = [(2, 2), (8, 3), (3, 6)]
    cov   = [[1, 0], [0, 1]]
    n_per_cluster = 200

    clusters = []
    labels   = []
    for i, mean in enumerate(means):
        pts = generate_gaussian_cluster(mean, cov, n_per_cluster, rng)
        clusters.append(pts)
        labels.append(np.full(n_per_cluster, i, dtype=int))

    X = np.vstack(clusters)
    y = np.concatenate(labels)
    return X, y


def generate_data_a2(seed=42):
    """
    Bài A2: 2400 điểm, 3 cụm lệch kích thước.
      - Cụm 0: 1200 điểm, tâm (2,2)
      - Cụm 1:  200 điểm, tâm (8,3)
      - Cụm 2: 1000 điểm, tâm (3,6)
    Cùng Σ = [[1,0],[0,1]].

    Returns:
        X (np.ndarray): shape (2400, 2)
        y (np.ndarray): Nhãn cụm thực (0, 1, 2), shape (2400,)
    """
    rng = np.random.default_rng(seed)

    means   = [(2, 2), (8, 3), (3, 6)]
    sizes   = [1200, 200, 1000]
    cov     = [[1, 0], [0, 1]]

    clusters = []
    labels   = []
    for i, (mean, n) in enumerate(zip(means, sizes)):
        pts = generate_gaussian_cluster(mean, cov, n, rng)
        clusters.append(pts)
        labels.append(np.full(n, i, dtype=int))

    X = np.vstack(clusters)
    y = np.concatenate(labels)
    return X, y


def generate_data_a3(seed=42):
    """
    Bài A3: 600 điểm, 3 cụm đều (200 điểm/cụm).
    Tâm: (2,2), (8,3), (3,6).
      - Cụm 0 và Cụm 1: Σ1 = [[1,0],[0,1]]  (hình cầu)
      - Cụm 2          : Σ2 = [[10,0],[0,1]] (kéo giãn theo trục X)

    Returns:
        X (np.ndarray): shape (600, 2)
        y (np.ndarray): Nhãn cụm thực (0, 1, 2), shape (600,)
    """
    rng = np.random.default_rng(seed)

    means        = [(2, 2), (8, 3), (3, 6)]
    cov_spherical = [[1,  0], [0, 1]]
    cov_stretched = [[10, 0], [0, 1]]
    n_per_cluster = 200

    covs = [cov_spherical, cov_spherical, cov_stretched]

    clusters = []
    labels   = []
    for i, (mean, cov) in enumerate(zip(means, covs)):
        pts = generate_gaussian_cluster(mean, cov, n_per_cluster, rng)
        clusters.append(pts)
        labels.append(np.full(n_per_cluster, i, dtype=int))

    X = np.vstack(clusters)
    y = np.concatenate(labels)
    return X, y


# ---------------------------------------------------------------------------
# Kiểm tra nhanh khi chạy trực tiếp file này
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Kiểm tra data_generator.py ===\n")

    X1, y1 = generate_data_a1()
    print(f"[A1] X shape: {X1.shape}, y shape: {y1.shape}")
    print(f"     Số điểm mỗi cụm: {[(y1==i).sum() for i in range(3)]}")

    X2, y2 = generate_data_a2()
    print(f"[A2] X shape: {X2.shape}, y shape: {y2.shape}")
    print(f"     Số điểm mỗi cụm: {[(y2==i).sum() for i in range(3)]}")

    X3, y3 = generate_data_a3()
    print(f"[A3] X shape: {X3.shape}, y shape: {y3.shape}")
    print(f"     Số điểm mỗi cụm: {[(y3==i).sum() for i in range(3)]}")
    print("\ndata_generator.py chạy thành công!")
