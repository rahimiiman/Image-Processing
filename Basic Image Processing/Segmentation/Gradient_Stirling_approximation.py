import numpy as np
import cv2
import matplotlib.pyplot as plt

# ---------------------------------
# Load grayscale image
# ---------------------------------
img = cv2.imread("test1.jpg", cv2.IMREAD_GRAYSCALE)
img = img.astype(np.float32)

rows, cols = img.shape

# =====================================================
# 1️⃣ Central Difference (2nd order)
# =====================================================
dx_c = np.zeros_like(img)
dy_c = np.zeros_like(img)

dx_c[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2.0
dy_c[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2.0

grad_c = np.sqrt(dx_c**2 + dy_c**2)

# =====================================================
# 2️⃣ Stirling (4th order)
# =====================================================
dx_s = np.zeros_like(img)
dy_s = np.zeros_like(img)

dx_s[:, 2:-2] = (
    -img[:, 4:] + 8*img[:, 3:-1]
    - 8*img[:, 1:-3] + img[:, :-4]
) / 12.0

dy_s[2:-2, :] = (
    -img[4:, :] + 8*img[3:-1, :]
    - 8*img[1:-3, :] + img[:-4, :]
) / 12.0

grad_s = np.sqrt(dx_s**2 + dy_s**2)

# =====================================================
# 3️⃣ Gradient of Gaussian
# =====================================================

# Gaussian smoothing first
sigma = 1.5
ksize = 7
img_blur = cv2.GaussianBlur(img, (ksize, ksize), sigma)

# Then central difference on smoothed image
dx_g = np.zeros_like(img)
dy_g = np.zeros_like(img)

dx_g[:, 1:-1] = (img_blur[:, 2:] - img_blur[:, :-2]) / 2.0
dy_g[1:-1, :] = (img_blur[2:, :] - img_blur[:-2, :]) / 2.0

grad_g = np.sqrt(dx_g**2 + dy_g**2)

# =====================================================
# Normalize for visualization
# =====================================================
def normalize(im):
    im = im - np.min(im)
    im = im / (np.max(im) + 1e-8)
    return im

grad_c_disp = normalize(grad_c)
grad_s_disp = normalize(grad_s)
grad_g_disp = normalize(grad_g)

# =====================================================
# Visualization
# =====================================================
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(grad_c_disp, cmap='gray')
plt.title("Central Difference")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(grad_s_disp, cmap='gray')
plt.title("Stirling (4th Order)")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(grad_g_disp, cmap='gray')
plt.title("Gradient of Gaussian")
plt.axis("off")

plt.tight_layout()
plt.show()