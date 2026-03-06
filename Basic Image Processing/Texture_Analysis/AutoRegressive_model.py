import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# Hyperparameters
# ==============================
input_path = "test.jpg"   # Set your image path
N = 9                      # AR window size (odd)
local_var_window = 21      # Window size for local residual variance

if N % 2 == 0:
    raise ValueError("N must be odd.")

# ==============================
# Load image
# ==============================
image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    raise ValueError("Image not found.")

image = image.astype(np.float64)
rows, cols = image.shape
half = N // 2

# ==============================
# Build AR system
# ==============================
X_list = []
y_list = []

for i in range(half, rows - half):
    for j in range(half, cols - half):
        window = image[i-half:i+half+1, j-half:j+half+1].flatten()
        center_idx = (N*N)//2
        y_list.append(window[center_idx])
        X_list.append(np.delete(window, center_idx))

X = np.array(X_list)
y = np.array(y_list)

# ==============================
# Estimate AR coefficients (OLS / BLUE under sigma^2 I)
# ==============================
a_hat = np.linalg.pinv(X) @ y
print("Estimated AR coefficients shape:", a_hat.shape)

# ==============================
# Predict image
# ==============================
y_pred = X @ a_hat

pred_img = np.zeros_like(image)
res_img = np.zeros_like(image)

idx = 0
for i in range(half, rows - half):
    for j in range(half, cols - half):
        pred_img[i, j] = y_pred[idx]
        res_img[i, j] = y[idx] - y_pred[idx]
        idx += 1

# ==============================
# Residual statistics
# ==============================
valid_res = res_img[half:rows-half, half:cols-half]

sigma2_hat = np.sum(valid_res**2) / (len(valid_res.flatten()) - len(a_hat))
print("Estimated innovation variance (sigma^2):", sigma2_hat)

# Residual energy map
res_energy = res_img**2

# ==============================
# Local innovation variance map
# ==============================
k = local_var_window
local_kernel = np.ones((k, k)) / (k*k)

local_var = cv2.filter2D(res_energy, -1, local_kernel)

# ==============================
# Visualization
# ==============================

plt.figure(figsize=(20,6))

# Original
plt.subplot(1,5,1)
plt.title("Original Image")
plt.imshow(image.astype(np.uint8), cmap='gray')
plt.axis('off')

# Predicted
plt.subplot(1,5,2)
plt.title("Predicted Image (AR)")
plt.imshow(np.clip(pred_img,0,255).astype(np.uint8), cmap='gray')
plt.axis('off')

# Signed residual
plt.subplot(1,5,3)
plt.title("Signed Residual")
im1 = plt.imshow(res_img, cmap='seismic')
plt.colorbar(im1, fraction=0.046)
plt.axis('off')

# Residual energy
plt.subplot(1,5,4)
plt.title("Residual Energy")
im2 = plt.imshow(res_energy, cmap='jet')
plt.colorbar(im2, fraction=0.046)
plt.axis('off')

# Local innovation variance
plt.subplot(1,5,5)
plt.title("Local Innovation Variance")
im3 = plt.imshow(local_var, cmap='hot')
plt.colorbar(im3, fraction=0.046)
plt.axis('off')

plt.tight_layout()
plt.show()