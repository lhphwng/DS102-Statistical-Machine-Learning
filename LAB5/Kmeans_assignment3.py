import time
import numpy as np

from data_generator import generate_data_a1, generate_data_a3
from kmeans import KMeansEM

# ---------------------------------------------------------------------------
# Cấu hình
# ---------------------------------------------------------------------------
K        = 3
MAX_ITER = 300
TOL      = 1e-4
N_RUNS   = 5

print("=" * 65)
print("  BÀI TẬP K-MEANS 3: Ảnh hưởng của Covariance Anisotropic")
print("=" * 65)

# ---------------------------------------------------------------------------
# Sinh dữ liệu
# ---------------------------------------------------------------------------
X_sph, y_sph  = generate_data_a1()   # Σ = I  (cầu - isotropic)
X_ani, y_ani  = generate_data_a3()   # Cụm 2: Σ = [[10,0],[0,1]] (kéo giãn)

print(f"\n[Dữ liệu SPHERICAL - A1]  Tất cả Σ = [[1,0],[0,1]]")
print(f"  N = {X_sph.shape[0]} | 200 pts/cụm | Tâm: (2,2),(8,3),(3,6)")

print(f"\n[Dữ liệu ANISOTROPIC - A3]")
print(f"  N = {X_ani.shape[0]} | 200 pts/cụm")
print(f"  Cụm 0 & 1: Σ = [[1,0],[0,1]]   (phân phối tròn)")
print(f"  Cụm 2    : Σ = [[10,0],[0,1]]  (kéo giãn mạnh theo trục X)")

# ---------------------------------------------------------------------------
# Hàm chạy K-Means nhiều lần, chọn nghiệm tốt nhất
# ---------------------------------------------------------------------------
def run_kmeans_best_of(X, k, n_runs, max_iter, tol, label=""):
    best_model   = None
    best_inertia = np.inf
    all_inertias = []
    all_times    = []

    for seed in range(n_runs):
        t0 = time.time()
        m  = KMeansEM(k=k, max_iter=max_iter, tol=tol, seed=seed)
        m.fit(X)
        t1 = time.time()
        all_times.append((t1 - t0) * 1000.0)
        all_inertias.append(m.inertia_)

        if m.inertia_ < best_inertia:
            best_inertia = m.inertia_
            best_model   = m

    return best_model, all_inertias, all_times

# ---------------------------------------------------------------------------
# Chạy trên SPHERICAL
# ---------------------------------------------------------------------------
print("\n" + "-" * 65)
print("[SPHERICAL - A1] Chạy K-Means...")
t0 = time.time()
model_sph, inertias_sph, times_sph = run_kmeans_best_of(
    X_sph, K, N_RUNS, MAX_ITER, TOL)
t1 = time.time()

print(f"  Tổng T. gian ({N_RUNS} lần): {(t1-t0)*1000:.2f} ms")
print(f"  T. gian TB / lần          : {np.mean(times_sph):.2f} ms")
print(f"  Inertia [min, mean, max]  : [{min(inertias_sph):.2f}, "
      f"{np.mean(inertias_sph):.2f}, {max(inertias_sph):.2f}]")
print(f"  Inertia (tốt nhất)        : {model_sph.inertia_:.4f}")
print(f"  Số vòng lặp               : {model_sph.n_iter_}")
print(f"  Tâm cụm tìm được:")
for j, c in enumerate(model_sph.centroids):
    print(f"    Cụm {j}: ({c[0]:.4f}, {c[1]:.4f})")

label_sph = [(model_sph.labels_==i).sum() for i in range(K)]
print(f"  Phân bố nhãn: {label_sph}")

# ---------------------------------------------------------------------------
# Chạy trên ANISOTROPIC
# ---------------------------------------------------------------------------
print("\n" + "-" * 65)
print("[ANISOTROPIC - A3] Chạy K-Means...")
t0 = time.time()
model_ani, inertias_ani, times_ani = run_kmeans_best_of(
    X_ani, K, N_RUNS, MAX_ITER, TOL)
t1 = time.time()

print(f"  Tổng T. gian ({N_RUNS} lần): {(t1-t0)*1000:.2f} ms")
print(f"  T. gian TB / lần          : {np.mean(times_ani):.2f} ms")
print(f"  Inertia [min, mean, max]  : [{min(inertias_ani):.2f}, "
      f"{np.mean(inertias_ani):.2f}, {max(inertias_ani):.2f}]")
print(f"  Inertia (tốt nhất)        : {model_ani.inertia_:.4f}")
print(f"  Số vòng lặp               : {model_ani.n_iter_}")
print(f"  Tâm cụm tìm được:")
for j, c in enumerate(model_ani.centroids):
    print(f"    Cụm {j}: ({c[0]:.4f}, {c[1]:.4f})")

label_ani = [(model_ani.labels_==i).sum() for i in range(K)]
print(f"  Phân bố nhãn: {label_ani}")

# ---------------------------------------------------------------------------
# Phân tích sai số tâm cụm
# ---------------------------------------------------------------------------
true_centers = np.array([[2.0, 2.0], [8.0, 3.0], [3.0, 6.0]])

def match_and_error(centroids, true_centers):
    """Ghép tâm tìm được với tâm thực theo khoảng cách nhỏ nhất."""
    used = set()
    errors = []
    for tc in true_centers:
        dists = [np.linalg.norm(centroids[j] - tc) if j not in used else 1e9
                 for j in range(len(centroids))]
        best_j = int(np.argmin(dists))
        used.add(best_j)
        errors.append(dists[best_j])
    return errors

err_sph = match_and_error(model_sph.centroids, true_centers)
err_ani = match_and_error(model_ani.centroids, true_centers)

print("\n[Sai số khoảng cách Euclidean so với tâm THỰC]")
print(f"  {'Cụm':>5} {'Tâm thực':>14} {'Err Spherical':>15} {'Err Anisotropic':>17}")
for i, (tc, e_s, e_a) in enumerate(zip(true_centers, err_sph, err_ani)):
    print(f"  {i:>5} ({tc[0]},{tc[1]}){' ':>6} {e_s:>14.4f} {e_a:>16.4f}")

print(f"\n  Tổng sai số (Spherical)   : {sum(err_sph):.4f}")
print(f"  Tổng sai số (Anisotropic) : {sum(err_ani):.4f}")

# ---------------------------------------------------------------------------
# NHẬN XÉT / BÌNH LUẬN
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("  NHẬN XÉT: Ảnh hưởng của Covariance Anisotropic đối với K-Means")
print("=" * 65)

print("""
[1] Giả định ngầm của K-Means - Khoảng cách Euclidean:
    Bước E của K-Means gán mỗi điểm dữ liệu x vào cụm k* sao cho:
        k* = argmin_k ||x - mu_k||^2
    Khoảng cách Euclidean này xem tất cả các chiều là BÌNH ĐẲNG và
    độc lập với nhau. Điều này tương đương với giả định rằng mỗi cụm
    có phân phối Gaussian với ma trận hiệp phương sai tỷ lệ với ma
    trận đơn vị: Σ_k = σ^2 * I (hình cầu, đẳng hướng).

[2] Vấn đề với Covariance kéo giãn (Anisotropic):
    Khi một cụm có Σ = [[10,0],[0,1]] (kéo giãn theo trục X với độ
    lệch chuẩn ~3.16 lần lớn hơn theo X so với Y), phân bố thực tế
    của cụm đó là một hình ELLIPSE trải dài theo trục X. Tuy nhiên,
    K-Means vẫn dùng khoảng cách Euclidean (vòng tròn) để phân vùng.

    Hệ quả:
    a) Tâm cụm bị kéo lệch: Các điểm ở biên của cụm ellipse (theo
       trục X) bị coi là "gần hơn" với tâm của các cụm lân cận khi
       đo bằng Euclidean, dẫn đến nhãn sai.
    b) Cụm ellipse bị cắt xén hoặc phân tách: K-Means có xu hướng
       "cắt ngang" cụm ellipse theo đường vuông góc với trục dài,
       chia nó thành 2 phần nhỏ hơn. Phần điểm nằm xa tâm (theo
       trục X) bị phân về cụm khác.
    c) Inertia tăng cao hơn: Do các điểm xa tâm (theo hướng kéo giãn)
       bị phân sai cụm, WCSS tổng thể cao hơn so với bài spherical.

[3] So sánh định lượng từ thí nghiệm:
    - Dữ liệu Spherical (A1): K-Means hoạt động chính xác, tâm tìm
      được rất gần tâm thực vì giả định Euclidean hoàn toàn phù hợp.
    - Dữ liệu Anisotropic (A3): Cụm 2 (tâm thực (3,6), Σ=[[10,0],[0,1]])
      gây ra sai số lớn nhất. Tâm cụm tìm được có thể bị lệch đáng
      kể so với tâm thực; một phần điểm cụm 2 bị gán nhầm sang cụm 0
      hoặc cụm 1.

[4] Đo lường bằng khoảng cách Mahalanobis vs Euclidean:
    Khoảng cách Mahalanobis: d_M(x, mu) = sqrt((x-mu)^T Σ^{-1} (x-mu))
    Khi Σ = [[10,0],[0,1]] thì Σ^{-1} = [[0.1,0],[0,1]], tức là trục
    X được "nén lại" khi tính khoảng cách. Nếu K-Means dùng khoảng
    cách Mahalanobis, nó sẽ phân cụm ellipse chính xác hơn nhiều.

[5] Mô hình phù hợp hơn cho dữ liệu anisotropic:
    - GMM với ma trận Σ đầy đủ (full covariance): Tự học hình dạng
      của từng cụm trong quá trình EM. GMM có thể xấp xỉ chính xác
      phân bố ellipse vì Bước M cập nhật Σ_k cho mỗi thành phần.
    - Spectral Clustering: Sử dụng đồ thị kết nối thay vì khoảng
      cách Euclidean, xử lý tốt hình dạng cụm phức tạp.
    - K-Means với Kernel (Kernel K-Means): Ánh xạ dữ liệu lên không
      gian đặc trưng phi tuyến trước khi phân cụm.

[6] Kết luận:
    Khoảng cách Euclidean là điểm mạnh nhưng cũng là điểm yếu cốt lõi
    của K-Means. Với dữ liệu có cấu trúc anisotropic (cụm ellipse, kéo
    giãn không đều), K-Means sẽ cho kết quả kém và khó cải thiện chỉ
    bằng cách điều chỉnh seed hay số vòng lặp. Lúc này, cần chuyển
    sang các thuật toán linh hoạt hơn như GMM.
""")

print("=" * 65)
print("  Kmeans_assignment3.py hoàn tất.")
print("=" * 65)
