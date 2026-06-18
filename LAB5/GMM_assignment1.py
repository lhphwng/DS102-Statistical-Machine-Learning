import time
import numpy as np

from data_generator import generate_data_a1
from gmm import GaussianMixtureModel
from kmeans import KMeansEM

# ---------------------------------------------------------------------------
# Cấu hình
# ---------------------------------------------------------------------------
K        = 3
MAX_ITER = 200
TOL      = 1e-4
N_RUNS   = 5    # Chạy nhiều lần để chọn nghiệm GMM tốt nhất

print("=" * 65)
print("  BÀI TẬP GMM 1: Mô hình xác suất & So sánh với K-Means")
print("=" * 65)

# ---------------------------------------------------------------------------
# Sinh dữ liệu
# ---------------------------------------------------------------------------
X, y_true = generate_data_a1()
N, D = X.shape

print(f"\n[Dữ liệu] N={N}, D={D}, K={K} cụm Gaussian đều nhau")
print(f"  Tâm thực: (2,2), (8,3), (3,6) | Σ = I\n")

# ---------------------------------------------------------------------------
# Chạy K-Means (làm baseline để so sánh)
# ---------------------------------------------------------------------------
print("=" * 65)
print("[BƯỚC 1] Chạy K-Means (Hard Assignment) làm baseline...")
print("-" * 65)

t_km_start = time.time()
best_km     = None
best_km_ine = np.inf

for seed in range(N_RUNS):
    km = KMeansEM(k=K, max_iter=300, tol=1e-4, seed=seed)
    km.fit(X)
    if km.inertia_ < best_km_ine:
        best_km_ine = km.inertia_
        best_km     = km

t_km_end = time.time()
t_km_ms  = (t_km_end - t_km_start) * 1000.0

print(f"  Thời gian ({N_RUNS} lần, chọn tốt nhất): {t_km_ms:.2f} ms")
print(f"  Inertia (WCSS) tốt nhất           : {best_km.inertia_:.4f}")
print(f"  Số vòng lặp                        : {best_km.n_iter_}")
print(f"  Tâm cụm (K-Means):")
for j, c in enumerate(best_km.centroids):
    print(f"    Cụm {j}: mu = ({c[0]:.4f}, {c[1]:.4f})")

# Thống kê hard assignment
km_labels  = best_km.labels_
km_dist    = [(km_labels==i).sum() for i in range(K)]
print(f"  Phân bố cụm (hard): {km_dist}")

# ---------------------------------------------------------------------------
# Chạy GMM (Soft Assignment)
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("[BƯỚC 2] Chạy GMM (Soft Assignment) bằng EM...")
print("-" * 65)

t_gmm_start  = time.time()
best_gmm     = None
best_gmm_ll  = -np.inf

for seed in range(N_RUNS):
    gmm = GaussianMixtureModel(K=K, max_iter=MAX_ITER, tol=TOL, seed=seed)
    gmm.fit(X)
    final_ll = gmm.log_likelihoods_[-1] if gmm.log_likelihoods_ else -np.inf
    if final_ll > best_gmm_ll:
        best_gmm_ll = final_ll
        best_gmm    = gmm

t_gmm_end = time.time()
t_gmm_ms  = (t_gmm_end - t_gmm_start) * 1000.0

print(f"  Thời gian ({N_RUNS} lần, chọn tốt nhất): {t_gmm_ms:.2f} ms")
print(f"  Log-Likelihood cuối (tốt nhất)     : {best_gmm_ll:.4f}")
print(f"  Số vòng lặp                        : {best_gmm.n_iter_}")

# ---------------------------------------------------------------------------
# In tham số GMM đã học được
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("[BƯỚC 3] Tham số GMM đã học được (pi, mu, Sigma):")
print("=" * 65)

for k in range(K):
    print(f"\n  --- Thành phần Gaussian k={k} ---")
    print(f"  pi[{k}]    = {best_gmm.pi[k]:.6f}  (trọng số / tỷ lệ cụm)")
    print(f"  mu[{k}]    = [{best_gmm.mu[k][0]:.6f}, {best_gmm.mu[k][1]:.6f}]")
    print(f"  sigma[{k}] =")
    s = best_gmm.sigma[k]
    print(f"    [[{s[0,0]:.6f}, {s[0,1]:.6f}],")
    print(f"     [{s[1,0]:.6f}, {s[1,1]:.6f}]]")

# Tổng kiểm tra: sum(pi) = 1
print(f"\n  Kiểm tra: sum(pi) = {best_gmm.pi.sum():.8f}  (phải = 1.0)")

# Soft assignment → hard label
gmm_labels = best_gmm.predict(X)
gmm_dist   = [(gmm_labels==i).sum() for i in range(K)]
print(f"\n  Phân bố cụm (GMM hard từ argmax R): {gmm_dist}")

# Responsibilities (soft) cho 5 điểm đầu
R_sample, _ = best_gmm._e_step(X[:5])
print("\n  Ví dụ Responsibilities (5 điểm đầu) - Soft Probabilities:")
print(f"  {'Điểm':>6} | {'r_k0':>10} | {'r_k1':>10} | {'r_k2':>10} | {'Label':>7}")
print("  " + "-" * 50)
for i in range(5):
    r = R_sample[i]
    lbl = np.argmax(r)
    print(f"  {i:>6} | {r[0]:>10.6f} | {r[1]:>10.6f} | {r[2]:>10.6f} | {lbl:>7}")

# ---------------------------------------------------------------------------
# So sánh sai số tâm cụm vs giá trị thực
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("[BƯỚC 4] So sánh tâm cụm học được vs giá trị thực:")
print("=" * 65)

true_centers = np.array([[2.0, 2.0], [8.0, 3.0], [3.0, 6.0]])
true_pi      = np.array([1/3, 1/3, 1/3])

def match_error(centers_found, true_centers):
    """Khớp tâm theo khoảng cách tối thiểu (greedy)."""
    used = set()
    errs = []
    order = []
    for tc in true_centers:
        best_d, best_j = np.inf, -1
        for j in range(len(centers_found)):
            if j not in used:
                d = np.linalg.norm(centers_found[j] - tc)
                if d < best_d:
                    best_d, best_j = d, j
        used.add(best_j)
        errs.append(best_d)
        order.append(best_j)
    return errs, order

km_errs, km_ord = match_error(best_km.centroids, true_centers)
gm_errs, gm_ord = match_error(best_gmm.mu, true_centers)

print(f"\n  {'Tâm thực':>14} | {'K-Means Err':>13} | {'GMM Err':>10}")
print("  " + "-" * 45)
for i, tc in enumerate(true_centers):
    print(f"  ({tc[0]},{tc[1]}){' ':>8} | {km_errs[i]:>13.6f} | {gm_errs[i]:>10.6f}")

print(f"\n  Tổng sai số - K-Means: {sum(km_errs):.6f}")
print(f"  Tổng sai số - GMM    : {sum(gm_errs):.6f}")

print(f"\n  [Pi GMM vs Thực tế]")
for k, j in enumerate(gm_ord):
    print(f"    Cụm {k}: pi_GMM = {best_gmm.pi[j]:.4f}  | pi_thực = {true_pi[k]:.4f}")

# ---------------------------------------------------------------------------
# NHẬN XÉT / BÌNH LUẬN
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("  NHẬN XÉT: So sánh Soft Assignment (GMM) vs Hard Assignment (K-Means)")
print("=" * 65)

print("""
[1] Bản chất Hard Assignment của K-Means:
    Trong K-Means, mỗi điểm dữ liệu x_n được gán DỨT KHOÁT (cứng)
    vào đúng một cụm duy nhất dựa trên khoảng cách Euclidean nhỏ nhất.
    Đây gọi là Hard Assignment hay Winner-Takes-All. Không có sự mơ
    hồ hay không chắc chắn nào được thể hiện trong phân công này.
    Hệ quả: K-Means ngầm giả định mỗi cụm là một Gaussian đẳng hướng
    với cùng phương sai, không thể biểu diễn sự chồng chéo giữa các
    cụm.

[2] Bản chất Soft Assignment của GMM:
    GMM dùng tham số responsibilities r_{nk} = P(z_n=k | x_n, theta)
    là xác suất hậu nghiệm để điểm x_n thuộc thành phần Gaussian k.
    Mỗi điểm được gán "mềm" (soft) vào TẤT CẢ các cụm với mức độ
    khác nhau. r_{nk} ∈ [0,1] và sum_k r_{nk} = 1. Điều này cho phép
    GMM:
    a) Biểu diễn sự KHÔNG CHẮC CHẮN: Điểm nằm giữa hai cụm sẽ có
       r_{n0} ≈ r_{n1} ≈ 0.5 thay vì bị gán cứng sang một bên.
    b) Mô hình hóa PHÂN PHỐI XÁC SUẤT đầy đủ: GMM cho p(x|theta) là
       hàm mật độ xác suất hợp lệ, có thể tính được P(x_mới) cho
       điểm dữ liệu mới (Density Estimation).
    c) Cụm hình dạng tùy ý: Ma trận Σ_k đầy đủ (full covariance) cho
       phép mô hình hóa cụm ellipse nghiêng bất kỳ.

[3] Ưu điểm của GMM so với K-Means:
    +-------------------------------+---------------+-------------------+
    | Tiêu chí                      | K-Means       | GMM               |
    +-------------------------------+---------------+-------------------+
    | Loại gán nhãn                 | Hard (0 or 1) | Soft (xác suất)   |
    | Hàm mục tiêu                  | WCSS ↓        | Log-Likelihood ↑  |
    | Hình dạng cụm                 | Chỉ cầu       | Ellipse tùy ý     |
    | Kích thước cụm lệch nhau      | Kém           | Tốt (qua pi_k)    |
    | Biểu diễn bất định            | Không         | Có                |
    | Density Estimation            | Không         | Có                |
    | Độ phức tạp tính toán         | Thấp          | Cao hơn           |
    | Nhạy cảm với local minima     | Cao           | Có nhưng ít hơn   |
    +-------------------------------+---------------+-------------------+

[4] Khi nào dùng GMM thay vì K-Means:
    - Khi cần ước lượng mật độ xác suất của dữ liệu (Anomaly Detection,
      Generative Modeling).
    - Khi dữ liệu có các cụm chồng chéo nhau (overlapping clusters).
    - Khi kích thước và hình dạng cụm không đều nhau.
    - Khi cần thông tin về độ tin cậy (uncertainty) của phân công cụm.
    - Khi dữ liệu thực sự có nguồn gốc từ hỗn hợp Gaussian (Gaussian
      Mixture generative model).

[5] Hạn chế của GMM:
    - Tốc độ tính toán chậm hơn K-Means, đặc biệt khi D và N lớn.
    - Nhạy cảm với giả định phân phối Gaussian; dữ liệu phi Gaussian
      sẽ cho kết quả kém.
    - Ma trận Σ_k đầy đủ có thể bị suy biến (singular) nếu không đủ
      dữ liệu hoặc không có regularization. Cần thêm epsilon vào
      đường chéo như đã cài đặt trong gmm.py.
    - Vẫn cần xác định K trước (tương tự K-Means).

[6] Kết luận từ thí nghiệm:
    Với dữ liệu bài A1 (3 cụm tách biệt, Σ = I), cả K-Means và GMM
    đều cho kết quả tốt. Tuy nhiên GMM học được thêm thông tin về
    trọng số pi và ma trận Σ của mỗi cụm, phản ánh chính xác hơn cấu
    trúc thống kê của dữ liệu. Đây là ưu điểm vượt trội của GMM khi
    áp dụng vào bài toán phức tạp hơn như phân đoạn ảnh (GMM 2).
""")

print("=" * 65)
print("  GMM_assignment1.py hoàn tất.")
print("=" * 65)
