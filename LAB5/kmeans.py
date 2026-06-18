import numpy as np

class KMeansEM:
    
    def __init__(self, k=3, max_iter=300, tol=1e-4, seed=None):
        self.k        = k
        self.max_iter = max_iter
        self.tol      = tol
        self.seed     = seed

        # Sẽ được gán sau khi fit()
        self.centroids = None
        self.labels_   = None
        self.inertia_  = None
        self.n_iter_   = 0

    # ------------------------------------------------------------------
    # Khởi tạo tâm cụm ngẫu nhiên
    # ------------------------------------------------------------------
    def _init_centroids(self, X):
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(X.shape[0], size=self.k, replace=False)
        return X[idx].copy()

    # ------------------------------------------------------------------
    # Bước E (Expectation): Gán nhãn cụm
    # ------------------------------------------------------------------
    def _e_step(self, X, centroids):
        # ||x||^2 : shape (N, 1)
        X_sq = np.sum(X ** 2, axis=1, keepdims=True)

        # ||c||^2 : shape (1, k)
        C_sq = np.sum(centroids ** 2, axis=1, keepdims=True).T

        # Cross term: 2 * X @ C.T : shape (N, k)
        cross = 2.0 * X @ centroids.T

        # Khoảng cách bình phương (N, k)
        dist_sq = X_sq + C_sq - cross

        # Gán nhãn cụm gần nhất
        labels = np.argmin(dist_sq, axis=1)
        return labels

    # ------------------------------------------------------------------
    # Bước M (Maximization): Cập nhật tâm cụm
    # ------------------------------------------------------------------
    def _m_step(self, X, labels):
        D = X.shape[1]
        new_centroids = np.zeros((self.k, D), dtype=float)

        for j in range(self.k):
            mask = (labels == j)
            if mask.sum() > 0:
                new_centroids[j] = X[mask].mean(axis=0)
            else:
                # Cụm rỗng: giữ nguyên tâm cũ
                new_centroids[j] = self.centroids[j]

        return new_centroids

    # ------------------------------------------------------------------
    # Tính WCSS / Inertia
    # ------------------------------------------------------------------
    def _compute_inertia(self, X, labels, centroids):
        diff = X - centroids[labels]           # (N, D)
        return float(np.sum(diff ** 2))

    # ------------------------------------------------------------------
    # Hàm fit() chính
    # ------------------------------------------------------------------
    def fit(self, X):
        X = np.array(X, dtype=float)

        # Khởi tạo tâm cụm ngẫu nhiên
        self.centroids = self._init_centroids(X)

        for iteration in range(1, self.max_iter + 1):
            # --- Bước E: Gán nhãn ---
            labels = self._e_step(X, self.centroids)

            # --- Bước M: Cập nhật tâm ---
            new_centroids = self._m_step(X, labels)

            # --- Kiểm tra hội tụ ---
            shift = np.max(np.linalg.norm(new_centroids - self.centroids, axis=1))
            self.centroids = new_centroids
            self.n_iter_   = iteration

            if shift < self.tol:
                break   # Hội tụ

        # Lưu kết quả cuối
        self.labels_  = self._e_step(X, self.centroids)
        self.inertia_ = self._compute_inertia(X, self.labels_, self.centroids)

        return self

    def predict(self, X):
        if self.centroids is None:
            raise RuntimeError("Mô hình chưa được huấn luyện. Hãy gọi fit() trước.")
        return self._e_step(np.array(X, dtype=float), self.centroids)


# ---------------------------------------------------------------------------
# Kiểm tra nhanh khi chạy trực tiếp file này
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data_generator import generate_data_a1

    print("=== Kiểm tra kmeans.py ===\n")
    X, y_true = generate_data_a1()

    model = KMeansEM(k=3, max_iter=300, tol=1e-4, seed=0)
    model.fit(X)

    print(f"Số vòng lặp hội tụ : {model.n_iter_}")
    print(f"Inertia (WCSS)      : {model.inertia_:.4f}")
    print(f"Tâm cụm cuối:\n{model.centroids}")
    print("\nkmeans.py chạy thành công!")
