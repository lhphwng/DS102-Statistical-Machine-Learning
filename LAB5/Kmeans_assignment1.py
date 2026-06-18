import time
import numpy as np

from data_generator import generate_data_a1
from kmeans import KMeansEM

# ---------------------------------------------------------------------------
# Cấu hình thí nghiệm
# ---------------------------------------------------------------------------
K         = 3      # Số cụm
N_RUNS    = 10     # Số lần chạy với hạt ngẫu nhiên khác nhau để quan sát local minima
MAX_ITER  = 300
TOL       = 1e-4

print("=" * 65)
print("  BÀI TẬP K-MEANS 1: Ảnh hưởng của khởi tạo ngẫu nhiên")
print("=" * 65)

# ---------------------------------------------------------------------------
# Sinh dữ liệu
# ---------------------------------------------------------------------------
X, y_true = generate_data_a1()
N, D = X.shape
print(f"\n[Dữ liệu] N={N} điểm, D={D} chiều, K={K} cụm thực.")
print(f"  Tâm thực: (2,2), (8,3), (3,6) | Σ = I (ma trận đơn vị)\n")

# ---------------------------------------------------------------------------
# Thí nghiệm: Chạy K-Means nhiều lần với seed khác nhau
# ---------------------------------------------------------------------------
print(f"{'Run':>4} {'Seed':>6} {'Inertia (WCSS)':>18} {'Vòng lặp':>10} {'T.gian (ms)':>14}")
print("-" * 60)

results = []

total_start = time.time()

for run in range(N_RUNS):
    seed = run * 7  # Các seed khác nhau

    t_start = time.time()
    model = KMeansEM(k=K, max_iter=MAX_ITER, tol=TOL, seed=seed)
    model.fit(X)
    t_end   = time.time()

    elapsed_ms = (t_end - t_start) * 1000.0
    results.append({
        'run'     : run + 1,
        'seed'    : seed,
        'inertia' : model.inertia_,
        'n_iter'  : model.n_iter_,
        'time_ms' : elapsed_ms,
        'centroids': model.centroids.copy(),
    })

    print(f"{run+1:>4} {seed:>6} {model.inertia_:>18.4f} {model.n_iter_:>10} {elapsed_ms:>13.2f}")

total_end = time.time()
total_ms  = (total_end - total_start) * 1000.0

# ---------------------------------------------------------------------------
# Tổng kết kết quả
# ---------------------------------------------------------------------------
inertias   = [r['inertia'] for r in results]
best_run   = results[int(np.argmin(inertias))]
worst_run  = results[int(np.argmax(inertias))]
mean_inert = float(np.mean(inertias))
std_inert  = float(np.std(inertias))

print("-" * 60)
print(f"\n[Tổng kết] Tổng thời gian {N_RUNS} lần chạy: {total_ms:.2f} ms")
print(f"  Inertia tốt nhất  (run {best_run['run']}, seed={best_run['seed']}): {best_run['inertia']:.4f}")
print(f"  Inertia tệ nhất   (run {worst_run['run']}, seed={worst_run['seed']}): {worst_run['inertia']:.4f}")
print(f"  Trung bình Inertia: {mean_inert:.4f} ± {std_inert:.4f}")
print(f"  Chênh lệch tốt/tệ : {worst_run['inertia'] - best_run['inertia']:.4f}")

print(f"\n  Tâm cụm của lần chạy tốt nhất (run {best_run['run']}):")
for j, c in enumerate(best_run['centroids']):
    print(f"    Cụm {j}: ({c[0]:.4f}, {c[1]:.4f})")

# ---------------------------------------------------------------------------
# NHẬN XÉT / BÌNH LUẬN (bắt buộc in bằng tiếng Việt)
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("  NHẬN XÉT: Ảnh hưởng của khởi tạo ngẫu nhiên đến K-Means")
print("=" * 65)

print("""
[1] Vấn đề hội tụ cục bộ (Local Minima) trong K-Means:
    K-Means tối thiểu hóa hàm mục tiêu WCSS (Within-Cluster Sum of
    Squares) thông qua vòng lặp EM. Tuy nhiên, hàm này KHÔNG LỒI
    (non-convex) đối với vị trí các tâm cụm, do đó thuật toán chỉ
    đảm bảo hội tụ đến nghiệm CỤC BỘ, không phải TOÀN CỤC.

[2] Ảnh hưởng trực tiếp của việc khởi tạo ngẫu nhiên:
    Mỗi lần chạy K-Means với một hạt ngẫu nhiên (seed) khác nhau
    sẽ chọn bộ tâm cụm ban đầu khác nhau. Nếu các tâm khởi tạo đặt
    ở vị trí xấu (ví dụ: nhiều tâm cùng nằm trong một cụm thực, hay
    không phủ đủ không gian dữ liệu), thuật toán sẽ bị "mắc kẹt" ở
    một local minimum với WCSS cao hơn so với nghiệm tối ưu toàn cục.

[3] Quan sát từ thí nghiệm này:
    - Với dữ liệu bài A1 (3 cụm tách biệt rõ ràng, kích thước đều
      nhau, Σ = I), K-Means thường hội tụ về cùng một nghiệm tốt cho
      phần lớn các seed vì cấu trúc dữ liệu đơn giản.
    - Khi chênh lệch Inertia giữa lần tốt nhất và tệ nhất đáng kể,
      đó là bằng chứng rõ ràng của vấn đề local minima.
    - Độ lệch chuẩn của Inertia phản ánh mức độ nhạy cảm của kết quả
      với việc khởi tạo: độ lệch chuẩn càng cao, bài toán càng khó.

[4] Giải pháp thực tế:
    a) K-Means++ (Arthur & Vassilvitskii, 2007): Khởi tạo thông minh
       bằng cách chọn mỗi tâm tiếp theo với xác suất tỷ lệ với bình
       phương khoảng cách đến tâm gần nhất đã chọn. Giảm đáng kể xác
       suất rơi vào local minima xấu.
    b) Chạy nhiều lần (Multiple Restarts): Thực hiện K-Means N lần
       với các seed khác nhau, chọn nghiệm có WCSS nhỏ nhất.
    c) EM với khởi tạo từ GMM: Sử dụng tâm cụm từ GMM (có soft
       assignment) làm điểm khởi đầu cho K-Means.

[5] Kết luận:
    Với dữ liệu đơn giản như bài A1, ảnh hưởng của local minima là
    nhỏ. Tuy nhiên, trên dữ liệu thực tế phức tạp hơn (cụm chồng
    nhau, kích thước lệch nhau, chiều cao), việc lựa chọn chiến lược
    khởi tạo tâm cụm là yếu tố THEN CHỐT quyết định chất lượng kết
    quả phân cụm của K-Means.
""")

print("=" * 65)
print("  Kmeans_assignment1.py hoàn tất.")
print("=" * 65)
