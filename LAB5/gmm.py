import numpy as np


# ---------------------------------------------------------------------------
# Hàm tính mật độ xác suất Gaussian đa biến (Multivariate Gaussian PDF)
# ---------------------------------------------------------------------------

def multivariate_gaussian_pdf(X, mu, sigma, eps=1e-6):
    D = X.shape[1]

    # Thêm epsilon vào đường chéo để đảm bảo ma trận không suy biến
    sigma_reg = sigma + eps * np.eye(D)

    # Định thức và ma trận nghịch đảo
    det_sigma = np.linalg.det(sigma_reg)
    inv_sigma = np.linalg.inv(sigma_reg)

    # Hằng số chuẩn hóa
    norm_const = 1.0 / (np.sqrt((2.0 * np.pi) ** D * np.abs(det_sigma)) + 1e-300)

    # Hiệu x - mu, shape (N, D)
    diff = X - mu

    # Mahalanobis distance bình phương: (N,) = diag[(N,D) @ (D,D) @ (D,N)]
    # Dùng einsum để tính hiệu quả: sum_j sum_l diff_ij * inv_jl * diff_il
    maha_sq = np.einsum('ij,jl,il->i', diff, inv_sigma, diff)

    # PDF
    pdf_vals = norm_const * np.exp(-0.5 * maha_sq)
    return pdf_vals


# ---------------------------------------------------------------------------
# Class GaussianMixtureModel
# ---------------------------------------------------------------------------

class GaussianMixtureModel:

    def __init__(self, K=3, max_iter=200, tol=1e-4, eps=1e-6, seed=None):
        self.K        = K
        self.max_iter = max_iter
        self.tol      = tol
        self.eps      = eps
        self.seed     = seed

        # Tham số mô hình (sau khi fit)
        self.pi    = None
        self.mu    = None
        self.sigma = None

        # Kết quả theo dõi
        self.log_likelihoods_ = []
        self.n_iter_          = 0

    # ------------------------------------------------------------------
    # Khởi tạo tham số
    # ------------------------------------------------------------------
    def _init_params(self, X):
        rng = np.random.default_rng(self.seed)
        N, D = X.shape

        # Trọng số đều nhau
        self.pi = np.full(self.K, 1.0 / self.K)

        # Chọn ngẫu nhiên K điểm làm mu ban đầu
        idx    = rng.choice(N, size=self.K, replace=False)
        self.mu = X[idx].copy()

        # Ma trận hiệp phương sai khởi tạo = Identity
        self.sigma = np.stack([np.eye(D)] * self.K, axis=0)  # (K, D, D)

    # ------------------------------------------------------------------
    # Bước E (Expectation): Tính responsibilities (xác suất hậu nghiệm)
    # ------------------------------------------------------------------
    def _e_step(self, X):
        N = X.shape[0]
        # Tính pi_k * pdf cho từng thành phần, shape (N, K)
        weighted_pdf = np.zeros((N, self.K))
        for k in range(self.K):
            pdf_k = multivariate_gaussian_pdf(X, self.mu[k], self.sigma[k], self.eps)
            weighted_pdf[:, k] = self.pi[k] * pdf_k

        # Tổng xác suất hỗn hợp, shape (N,)
        pdf_sum = weighted_pdf.sum(axis=1, keepdims=True)

        # Tránh chia cho 0
        pdf_sum = np.maximum(pdf_sum, 1e-300)

        # Responsibilities
        R = weighted_pdf / pdf_sum

        # Log-Likelihood: log p(X) = sum_n log(sum_k pi_k * N(x_n|...))
        log_likelihood = np.sum(np.log(pdf_sum + 1e-300))

        return R, float(log_likelihood)

    # ------------------------------------------------------------------
    # Bước M (Maximization): Cập nhật tham số pi, mu, sigma
    # ------------------------------------------------------------------
    def _m_step(self, X, R):
        N, D = X.shape

        for k in range(self.K):
            r_k = R[:, k]            # shape (N,)
            N_k = r_k.sum() + 1e-300  # Tránh chia cho 0

            # Cập nhật pi
            self.pi[k] = N_k / N

            # Cập nhật mu
            self.mu[k] = (r_k[:, np.newaxis] * X).sum(axis=0) / N_k

            # Cập nhật sigma
            diff = X - self.mu[k]                       # (N, D)
            # sigma_k = sum_n r_nk * diff_n @ diff_n.T / N_k
            # Dùng einsum: r_k (N,) * diff (N,D) * diff (N,D) -> (D,D)
            weighted_outer = np.einsum('n,ni,nj->ij', r_k, diff, diff)
            self.sigma[k]  = weighted_outer / N_k

    # ------------------------------------------------------------------
    # Hàm fit() chính
    # ------------------------------------------------------------------
    def fit(self, X):
        X = np.array(X, dtype=float)

        # Khởi tạo tham số
        self._init_params(X)
        self.log_likelihoods_ = []
        prev_ll = -np.inf

        for iteration in range(1, self.max_iter + 1):
            # --- Bước E ---
            R, log_likelihood = self._e_step(X)
            self.log_likelihoods_.append(log_likelihood)

            # --- Kiểm tra hội tụ ---
            if abs(log_likelihood - prev_ll) < self.tol:
                self.n_iter_ = iteration
                break

            prev_ll = log_likelihood

            # --- Bước M ---
            self._m_step(X, R)

        else:
            self.n_iter_ = self.max_iter

        return self

    # ------------------------------------------------------------------
    # Dự đoán nhãn cụm (hard assignment từ soft probabilities)
    # ------------------------------------------------------------------
    def predict(self, X):
        if self.mu is None:
            raise RuntimeError("Mô hình chưa được huấn luyện. Hãy gọi fit() trước.")
        R, _ = self._e_step(np.array(X, dtype=float))
        return np.argmax(R, axis=1)

    def predict_proba(self, X):
        if self.mu is None:
            raise RuntimeError("Mô hình chưa được huấn luyện. Hãy gọi fit() trước.")
        R, _ = self._e_step(np.array(X, dtype=float))
        return R


# ---------------------------------------------------------------------------
# Kiểm tra nhanh khi chạy trực tiếp file này
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data_generator import generate_data_a1

    print("=== Kiểm tra gmm.py ===\n")
    X, y_true = generate_data_a1()

    gmm = GaussianMixtureModel(K=3, max_iter=100, tol=1e-4, seed=0)
    gmm.fit(X)

    print(f"Số vòng lặp hội tụ : {gmm.n_iter_}")
    print(f"Log-Likelihood cuối : {gmm.log_likelihoods_[-1]:.4f}")
    print(f"Trọng số pi         : {gmm.pi}")
    print(f"Tâm mu:\n{gmm.mu}")
    print("\ngmm.py chạy thành công!")
