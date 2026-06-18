import time
import numpy as np

from data_generator import generate_data_a1, generate_data_a2
from kmeans import KMeansEM

# ---------------------------------------------------------------------------
# Cấu hình
# ---------------------------------------------------------------------------
K        = 3
MAX_ITER = 300
TOL      = 1e-4
N_RUNS   = 5    # Chạy nhiều lần để loại trừ ảnh hưởng seed

print("=" * 65)
print("  BÀI TẬP K-MEANS 2: Ảnh hưởng của kích thước cụm lệch nhau")
print("=" * 65)

# ---------------------------------------------------------------------------
# Sinh cả hai tập dữ liệu để so sánh
# ---------------------------------------------------------------------------
X_bal,  y_bal  = generate_data_a1()  # Cân bằng: 200/200/200
X_imbal, y_imbal = generate_data_a2()  # Lệch: 1200/200/1000

print(f"\n[Dữ liệu BALANCED - A1]")
print(f"  N = {X_bal.shape[0]} điểm | Phân bố: {[(y_bal==i).sum() for i in range(3)]}")

print(f"\n[Dữ liệu IMBALANCED - A2]")
print(f"  N = {X_imbal.shape[0]} điểm | Phân bố: {[(y_imbal==i).sum() for i in range(3)]}")
print(f"  Tâm: (2,2)=1200 pts, (8,3)=200 pts, (3,6)=1000 pts | Σ = I")

# ---------------------------------------------------------------------------
# Hàm chạy K-Means nhiều lần và chọn nghiệm tốt nhất
# ---------------------------------------------------------------------------
def run_kmeans_best_of(X, k, n_runs, max_iter, tol):
    """Chạy K-Means n_runs lần, trả về nghiệm có Inertia thấp nhất."""
    best_model  = None
    best_inertia = np.inf
    times_ms    = []

    for seed in range(n_runs):
        t0 = time.time()
        m  = KMeansEM(k=k, max_iter=max_iter, tol=tol, seed=seed)
        m.fit(X)
        t1 = time.time()
        times_ms.append((t1 - t0) * 1000.0)

        if m.inertia_ < best_inertia:
            best_inertia = m.inertia_
            best_model   = m

    return best_model, times_ms

# ---------------------------------------------------------------------------
# Chạy trên dữ liệu BALANCED
# ---------------------------------------------------------------------------
print("\n" + "-" * 65)
print("[BALANCED] Chạy K-Means...")
t_bal_start = time.time()
model_bal, times_bal = run_kmeans_best_of(X_bal, K, N_RUNS, MAX_ITER, TOL)
t_bal_end   = time.time()

print(f"  Tổng thời gian ({N_RUNS} lần): {(t_bal_end - t_bal_start)*1000:.2f} ms")
print(f"  Thời gian trung bình / lần  : {np.mean(times_bal):.2f} ms")
print(f"  Inertia tốt nhất            : {model_bal.inertia_:.4f}")
print(f"  Số vòng lặp (tốt nhất)      : {model_bal.n_iter_}")
print(f"  Tâm cụm tìm được:")
for j, c in enumerate(model_bal.centroids):
    print(f"    Cụm {j}: ({c[0]:.4f}, {c[1]:.4f})")

# Phân bố nhãn tìm được
label_dist_bal = [(model_bal.labels_==i).sum() for i in range(K)]
print(f"  Phân bố nhãn tìm được      : {label_dist_bal}")

# ---------------------------------------------------------------------------
# Chạy trên dữ liệu IMBALANCED
# ---------------------------------------------------------------------------
print("\n" + "-" * 65)
print("[IMBALANCED] Chạy K-Means...")
t_imbal_start = time.time()
model_imbal, times_imbal = run_kmeans_best_of(X_imbal, K, N_RUNS, MAX_ITER, TOL)
t_imbal_end   = time.time()

print(f"  Tổng thời gian ({N_RUNS} lần): {(t_imbal_end - t_imbal_start)*1000:.2f} ms")
print(f"  Thời gian trung bình / lần  : {np.mean(times_imbal):.2f} ms")
print(f"  Inertia tốt nhất            : {model_imbal.inertia_:.4f}")
print(f"  Số vòng lặp (tốt nhất)      : {model_imbal.n_iter_}")
print(f"  Tâm cụm tìm được:")
for j, c in enumerate(model_imbal.centroids):
    print(f"    Cụm {j}: ({c[0]:.4f}, {c[1]:.4f})")

label_dist_imbal = [(model_imbal.labels_==i).sum() for i in range(K)]
print(f"  Phân bố nhãn tìm được      : {label_dist_imbal}")

# ---------------------------------------------------------------------------
# So sánh tâm cụm tìm được vs thực tế
# ---------------------------------------------------------------------------
print("\n[So sánh tâm cụm thực vs tìm được]")
true_centers = np.array([[2, 2], [8, 3], [3, 6]])
print(f"  Tâm THỰC: {true_centers.tolist()}")

# Sắp xếp tâm theo trục x để so sánh trực quan
sorted_bal   = model_bal.centroids[np.argsort(model_bal.centroids[:, 0])]
sorted_imbal = model_imbal.centroids[np.argsort(model_imbal.centroids[:, 0])]
print(f"  Balanced  (sắp x): {np.round(sorted_bal, 4).tolist()}")
print(f"  Imbalanced(sắp x): {np.round(sorted_imbal, 4).tolist()}")

# ---------------------------------------------------------------------------
# NHẬN XÉT / BÌNH LUẬN
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("  NHẬN XÉT: Ảnh hưởng của kích thước cụm lệch nhau (Imbalanced)")
print("=" * 65)

print("""
[1] Cơ chế hoạt động của K-Means và vấn đề lệch kích thước:
    K-Means tối thiểu hóa WCSS bằng cách gán mỗi điểm vào cụm có
    TÂM GẦN NHẤT theo khoảng cách Euclidean. Thuật toán này KHÔNG có
    tham số tường minh nào phản ánh kích thước (hay trọng số) của từng
    cụm - tất cả các cụm được đối xử BÌNH ĐẲNG về mặt trọng số.

[2] Hiệu ứng quan sát được khi kích thước cụm lệch lớn:
    a) "Hút" tâm cụm về phía cụm lớn:
       Tâm của cụm lớn (1200 điểm, tâm (2,2)) bị kéo lệch về phía
       trung tâm hình học của cụm, nhưng ít bị ảnh hưởng hơn so với
       trường hợp cụm nhỏ. Ngược lại, cụm nhỏ (200 điểm, tâm (8,3))
       dễ bị "lấn át" nếu nằm gần cụm lớn.

    b) Chia nhỏ cụm lớn thành nhiều cụm giả (over-segmentation):
       Khi tổng số cụm K cố định và có một cụm rất lớn, K-Means có
       xu hướng tách cụm đó thành 2 hoặc nhiều cụm con để giảm WCSS,
       trong khi bỏ qua cụm nhỏ thực sự.

    c) Ranh giới quyết định (Decision Boundary) bị lệch:
       Ranh giới phân chia trong K-Means là đường thẳng trung bình
       vuông góc (perpendicular bisector) giữa hai tâm. Ranh giới này
       KHÔNG phụ thuộc vào kích thước cụm. Do đó, một cụm nhỏ có thể
       bị "nuốt" bởi cụm lớn bên cạnh nếu các tâm không đủ xa nhau.

[3] So sánh thực nghiệm Balanced vs Imbalanced:
    - Với dữ liệu balanced (A1): 3 cụm đều 200 điểm, K-Means hội tụ
      nhanh và tìm được các tâm rất gần với giá trị thực.
    - Với dữ liệu imbalanced (A2): Cụm nhỏ 200 điểm (8,3) nằm giữa
      hai cụm lớn, dễ dẫn đến việc K-Means có thể phân bổ sai một
      phần điểm của cụm nhỏ sang cụm lớn lân cận.
    - Thời gian chạy trên A2 lâu hơn do N lớn hơn (2400 vs 600).

[4] Hạn chế cốt lõi của K-Means:
    K-Means ngầm giả định rằng các cụm có KÍCH THƯỚC TƯƠNG ĐƯƠNG và
    HÌNH DẠNG CẦU (Euclidean). Khi giả định này bị vi phạm, thuật
    toán sẽ cho kết quả kém chính xác.

[5] Giải pháp thay thế khi gặp imbalanced clusters:
    - GMM với tham số pi (trọng số): Mô hình hóa tường minh xác suất
      prior của mỗi cụm, giúp xử lý lệch kích thước tốt hơn.
    - Weighted K-Means: Gán trọng số khác nhau cho từng điểm dữ liệu.
    - DBSCAN: Phân cụm dựa trên mật độ, không cần xác định K trước,
      xử lý tốt cụm có kích thước và hình dạng đa dạng.
""")

print("=" * 65)
print("  Kmeans_assignment2.py hoàn tất.")
print("=" * 65)
