import time
import numpy as np
import cv2
import os

from gmm import GaussianMixtureModel

# ---------------------------------------------------------------------------
# CẤU HÌNH ĐƯỜNG DẪN FILE ẢNH
# Thay đổi INPUT_IMAGE_PATH để trỏ đến file ảnh cow.jpg của bạn.
# Script cần chạy trong cùng thư mục với file ảnh, hoặc đặt đường dẫn đầy đủ.
# ---------------------------------------------------------------------------
INPUT_IMAGE_PATH  = "cow.jpg"
OUTPUT_IMAGE_PATH = "output_image.jpg"
OUTPUT_MASK_PATH  = "output_mask.jpg"    # Mặt nạ nhị phân (debug)
OUTPUT_FG_PATH    = "output_foreground.jpg"  # Chỉ giữ foreground (đối tượng)

# Tham số GMM
K_CLUSTERS = 3      # Số cụm màu. K=2 cho ảnh đơn giản, K=3 cho ảnh phức tạp hơn
MAX_ITER   = 100
TOL        = 1e-3
N_RUNS     = 3      # Chạy nhiều lần, chọn LL tốt nhất
GMM_SEED   = 42

print("=" * 65)
print("  BÀI TẬP GMM 2: Tách Nền Ảnh (Background Subtraction)")
print("=" * 65)
print(f"\n  Ảnh đầu vào : {INPUT_IMAGE_PATH}")
print(f"  Số cụm K    : {K_CLUSTERS}")
print(f"  Ảnh đầu ra  : {OUTPUT_IMAGE_PATH}")

# ---------------------------------------------------------------------------
# BƯỚC 1: Đọc ảnh bằng OpenCV
# ---------------------------------------------------------------------------
print("\n" + "-" * 65)
print("[BƯỚC 1] Đọc ảnh bằng OpenCV...")

if not os.path.isfile(INPUT_IMAGE_PATH):
    print(f"\n  [LỖI] Không tìm thấy file ảnh: '{INPUT_IMAGE_PATH}'")
    print("  Vui lòng đặt file 'cow.jpg' vào cùng thư mục với script này,")
    print("  hoặc thay đổi biến INPUT_IMAGE_PATH thành đường dẫn đúng.")
    raise FileNotFoundError(f"Không tìm thấy: {INPUT_IMAGE_PATH}")

# cv2.imread đọc ảnh theo thứ tự kênh màu BGR
img_bgr = cv2.imread(INPUT_IMAGE_PATH)
if img_bgr is None:
    raise ValueError(f"OpenCV không thể đọc file ảnh: {INPUT_IMAGE_PATH}")

H, W, C = img_bgr.shape
print(f"  Kích thước ảnh: {W} x {H} pixels, {C} kênh màu (BGR)")
print(f"  Tổng số pixels : {H * W}")

# ---------------------------------------------------------------------------
# BƯỚC 2: Chuyển BGR → RGB và reshape thành (N_pixels, 3)
# ---------------------------------------------------------------------------
print("\n[BƯỚC 2] Tiền xử lý ảnh...")

# Chuyển hệ màu BGR → RGB
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
print(f"  Đã chuyển BGR → RGB")

# Reshape (H, W, 3) → (N_pixels, 3)
N_pixels = H * W
pixels   = img_rgb.reshape(N_pixels, 3).astype(np.float64)
print(f"  Reshape: ({H}, {W}, 3) → ({N_pixels}, 3)")

# Chuẩn hóa giá trị pixel về [0, 1] để EM ổn định hơn
pixels_norm = pixels / 255.0
print(f"  Chuẩn hóa pixel về [0.0, 1.0]")
print(f"  Min pixel: {pixels_norm.min():.4f}, Max pixel: {pixels_norm.max():.4f}")

# ---------------------------------------------------------------------------
# BƯỚC 3: Khởi tạo và huấn luyện GMM
# ---------------------------------------------------------------------------
print("\n" + "-" * 65)
print(f"[BƯỚC 3] Huấn luyện GMM với K={K_CLUSTERS} cụm...")
print(f"  Chạy {N_RUNS} lần, chọn nghiệm có Log-Likelihood cao nhất.")

best_gmm    = None
best_ll     = -np.inf
t_total_start = time.time()

for run in range(N_RUNS):
    seed = GMM_SEED + run * 13
    print(f"\n  --- Run {run+1}/{N_RUNS} (seed={seed}) ---")

    t_run_start = time.time()
    gmm = GaussianMixtureModel(
        K        = K_CLUSTERS,
        max_iter = MAX_ITER,
        tol      = TOL,
        eps      = 1e-6,
        seed     = seed
    )
    gmm.fit(pixels_norm)
    t_run_end   = time.time()

    final_ll = gmm.log_likelihoods_[-1] if gmm.log_likelihoods_ else -np.inf
    t_run_ms = (t_run_end - t_run_start) * 1000.0

    print(f"  Vòng lặp       : {gmm.n_iter_}")
    print(f"  Log-Likelihood : {final_ll:.4f}")
    print(f"  Thời gian      : {t_run_ms:.2f} ms")

    if final_ll > best_ll:
        best_ll  = final_ll
        best_gmm = gmm
        print(f"  ✓ Nghiệm tốt nhất cập nhật (LL={best_ll:.4f})")

t_total_end = time.time()
t_total_ms  = (t_total_end - t_total_start) * 1000.0

print(f"\n[Tổng kết GMM]")
print(f"  Tổng thời gian EM ({N_RUNS} lần): {t_total_ms:.2f} ms")
print(f"  Log-Likelihood tốt nhất       : {best_ll:.4f}")

# ---------------------------------------------------------------------------
# BƯỚC 4: Phân tích các cụm GMM học được
# ---------------------------------------------------------------------------
print("\n" + "-" * 65)
print("[BƯỚC 4] Phân tích tham số GMM đã học...")

print(f"\n  {'Cụm k':>6} | {'pi_k':>8} | {'mu_R':>7} | {'mu_G':>7} | {'mu_B':>7} | {'Màu (0-255)':>14}")
print("  " + "-" * 65)

for k in range(K_CLUSTERS):
    pi_k = best_gmm.pi[k]
    mu_k = best_gmm.mu[k] * 255.0  # Scale về 0-255 để dễ đọc
    print(f"  {k:>6} | {pi_k:>8.4f} | {mu_k[0]:>7.1f} | {mu_k[1]:>7.1f} | {mu_k[2]:>7.1f} | R={mu_k[0]:.0f} G={mu_k[1]:.0f} B={mu_k[2]:.0f}")

# ---------------------------------------------------------------------------
# BƯỚC 5: Xác định cụm nền (background cluster)
# ---------------------------------------------------------------------------
print("\n" + "-" * 65)
print("[BƯỚC 5] Xác định cụm nền (Background)...")

# Chiến lược: Cụm có trọng số pi lớn nhất thường là nền vì nền
# thường chiếm diện tích lớn nhất trong ảnh.
# Ngoài ra, ảnh cow.jpg thường có nền là bầu trời hoặc đồng cỏ
# (các màu sáng / trung tính). Cụm có pi cao nhất là ứng viên nền.
bg_cluster_idx = int(np.argmax(best_gmm.pi))

print(f"  Chiến lược     : Cụm có trọng số pi lớn nhất = nền")
print(f"  Cụm nền        : k = {bg_cluster_idx}")
print(f"  pi (nền)       : {best_gmm.pi[bg_cluster_idx]:.4f}")
mu_bg = best_gmm.mu[bg_cluster_idx] * 255.0
print(f"  Màu trung bình : R={mu_bg[0]:.1f}, G={mu_bg[1]:.1f}, B={mu_bg[2]:.1f}")
print(f"  (Cụm này chiếm ~{best_gmm.pi[bg_cluster_idx]*100:.1f}% ảnh)")

# ---------------------------------------------------------------------------
# BƯỚC 6: Tính Responsibilities và tạo Binary Mask
# ---------------------------------------------------------------------------
print("\n" + "-" * 65)
print("[BƯỚC 6] Tính Responsibilities và tạo Binary Mask...")

# Lấy xác suất hậu nghiệm (soft assignment), shape (N_pixels, K)
R_all, _ = best_gmm._e_step(pixels_norm)
print(f"  Responsibilities shape: {R_all.shape}")

# Gán nhãn cứng dựa trên argmax
labels_pixels = np.argmax(R_all, axis=1)   # shape (N_pixels,)

# Tạo binary mask: 0 = nền, 255 = foreground
bg_mask_flat = (labels_pixels == bg_cluster_idx).astype(np.uint8)  # 1 = nền

# Binary mask dạng ảnh (255=nền, 0=đối tượng) → dùng để xem nền
mask_bg     = (bg_mask_flat * 255).reshape(H, W)                    # nền trắng
# Foreground mask (255=đối tượng, 0=nền)
mask_fg     = ((1 - bg_mask_flat) * 255).reshape(H, W)

n_bg = bg_mask_flat.sum()
n_fg = N_pixels - n_bg
print(f"  Số pixel nền        : {n_bg} ({n_bg/N_pixels*100:.1f}%)")
print(f"  Số pixel foreground : {n_fg} ({n_fg/N_pixels*100:.1f}%)")

# ---------------------------------------------------------------------------
# BƯỚC 7: Tạo ảnh kết quả và lưu file
# ---------------------------------------------------------------------------
print("\n" + "-" * 65)
print("[BƯỚC 7] Tạo ảnh kết quả và lưu file...")

# --- Ảnh kết quả chính: Ảnh gốc với nền bị tô đen ---
# Chuyển foreground mask về 3 kênh
mask_fg_3ch = np.stack([mask_fg, mask_fg, mask_fg], axis=2)  # (H, W, 3)

# Giữ pixel foreground, tô đen pixel nền
img_result_rgb = img_rgb.copy()
img_result_rgb[mask_fg == 0] = 0    # Tô đen vùng nền

# Chuyển về BGR để lưu bằng cv2
img_result_bgr = cv2.cvtColor(img_result_rgb, cv2.COLOR_RGB2BGR)
cv2.imwrite(OUTPUT_IMAGE_PATH, img_result_bgr)
print(f"  ✓ Lưu ảnh kết quả (nền tô đen)  → '{OUTPUT_IMAGE_PATH}'")

# --- Lưu Binary Mask (foreground mask) ---
cv2.imwrite(OUTPUT_MASK_PATH, mask_fg)
print(f"  ✓ Lưu binary mask (trắng=FG)      → '{OUTPUT_MASK_PATH}'")

# --- Lưu ảnh chỉ có foreground với nền trắng ---
img_fg_white_rgb = img_rgb.copy()
img_fg_white_rgb[mask_fg == 0] = 255    # Tô TRẮNG vùng nền
img_fg_white_bgr = cv2.cvtColor(img_fg_white_rgb, cv2.COLOR_RGB2BGR)
cv2.imwrite(OUTPUT_FG_PATH, img_fg_white_bgr)
print(f"  ✓ Lưu ảnh foreground (nền trắng)  → '{OUTPUT_FG_PATH}'")

# Kiểm tra file đã lưu
for fpath in [OUTPUT_IMAGE_PATH, OUTPUT_MASK_PATH, OUTPUT_FG_PATH]:
    size_kb = os.path.getsize(fpath) / 1024
    print(f"    [{fpath}] kích thước: {size_kb:.1f} KB")

# ---------------------------------------------------------------------------
# BƯỚC 8: Thống kê phân cụm màu
# ---------------------------------------------------------------------------
print("\n" + "-" * 65)
print("[BƯỚC 8] Thống kê phân cụm màu GMM cuối cùng:")

labels_2d = labels_pixels.reshape(H, W)

for k in range(K_CLUSTERS):
    n_k     = (labels_pixels == k).sum()
    pct_k   = n_k / N_pixels * 100
    mu_rgb  = best_gmm.mu[k] * 255.0
    role    = "NỀN (Background)" if k == bg_cluster_idx else "Đối tượng (Foreground)"
    print(f"  Cụm {k}: {n_k:>7} px ({pct_k:5.1f}%) | "
          f"mu_RGB=({mu_rgb[0]:.0f},{mu_rgb[1]:.0f},{mu_rgb[2]:.0f}) | {role}")

# ---------------------------------------------------------------------------
# NHẬN XÉT / BÌNH LUẬN
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("  NHẬN XÉT: Ứng dụng GMM trong Tách Nền Ảnh")
print("=" * 65)

print(f"""
[1] Nguyên lý ứng dụng GMM cho phân đoạn màu ảnh:
    Mỗi pixel trong ảnh màu RGB được xem là một vector 3 chiều
    x = (R, G, B) ∈ [0,255]^3. Toàn bộ N={N_pixels} pixel của ảnh
    {W}x{H} tạo thành tập dữ liệu trong không gian màu 3D.
    GMM mô hình hóa phân phối màu của toàn bộ ảnh như một hỗn hợp
    K={K_CLUSTERS} phân phối Gaussian:
        p(x) = sum_k pi_k * N(x | mu_k, Sigma_k)
    Mỗi Gaussian đại diện cho một "vùng màu" chính trong ảnh.

[2] Chiến lược tách nền:
    Cụm có TRỌNG SỐ pi lớn nhất được chọn làm nền vì:
    - Nền ảnh thường chiếm diện tích lớn nhất (đồng cỏ, bầu trời...).
    - pi_k tỷ lệ với số lượng pixel thuộc cụm k sau khi EM hội tụ.
    - Cụm nền k*={bg_cluster_idx} với pi={best_gmm.pi[bg_cluster_idx]:.4f}
      chiếm ~{best_gmm.pi[bg_cluster_idx]*100:.1f}% tổng số pixel.

[3] Ưu điểm của GMM so với phương pháp ngưỡng đơn giản:
    - Phân đoạn MỀM (Soft Segmentation): Pixel ở vùng biên (giữa đối
      tượng và nền) được gán xác suất thay vì nhị phân cứng, cho phép
      xử lý tinh tế hơn (anti-aliasing).
    - Học hình dạng phân phối màu: Sigma_k đầy đủ nắm bắt tương quan
      giữa kênh R, G, B trong mỗi vùng (ví dụ: vùng da người có
      tương quan R↑↑ G↑ B↓).
    - Không cần đặt ngưỡng thủ công: EM tự động tìm ranh giới màu
      tối ưu theo dữ liệu thực tế của ảnh.

[4] Hạn chế và hướng cải thiện:
    a) Nhạy cảm với K: Nếu K quá nhỏ, cụm không đủ để phân biệt nền
       và đối tượng. Nếu K quá lớn, nền bị chia thành nhiều cụm nhỏ.
    b) Chỉ dùng thông tin màu: GMM không xét thông tin không gian
       (vị trí pixel), dẫn đến mặt nạ "bốc" (noisy). Cải thiện:
       dùng thêm vị trí (R, G, B, x_norm, y_norm) làm đặc trưng.
    c) Nền phức tạp (nhiều màu): Nên tăng K=4 hoặc K=5, hoặc kết hợp
       GrabCut (dùng GMM trong vòng lặp với user interaction).
    d) Thời gian chạy: O(N_pixels * K * D^2) mỗi vòng lặp EM. Trên
       ảnh lớn có thể chậm → cải thiện bằng mini-batch EM hoặc
       downsample ảnh khi train.

[5] File đầu ra đã được tạo:
    - {OUTPUT_IMAGE_PATH}  : Ảnh gốc, nền tô màu đen.
    - {OUTPUT_MASK_PATH}   : Binary mask (trắng=foreground, đen=nền).
    - {OUTPUT_FG_PATH}     : Đối tượng trên nền trắng.
""")

print("=" * 65)
print("  GMM_assignment2.py hoàn tất.")
print("=" * 65)
