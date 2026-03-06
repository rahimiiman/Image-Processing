import numpy as np
import cv2
import matplotlib.pyplot as plt

# ---------------------------------
# Load grayscale image
# ---------------------------------
img = cv2.imread("test4.jpg", cv2.IMREAD_GRAYSCALE)
img = img.astype(np.float32)

rows, cols = img.shape

# ---------------------------------
# Allocate derivative arrays
# ---------------------------------
dx = np.zeros_like(img)
dy = np.zeros_like(img)

# ---------------------------------
# Central Difference (Symmetric)
# ---------------------------------

# d/dx  -> horizontal derivative
dx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2.0

# d/dy  -> vertical derivative
dy[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2.0

# ---------------------------------
# Gradient magnitude
# ---------------------------------
grad_mag = np.sqrt(dx**2 + dy**2)

# ---------------------------------
# Normalize for visualization
# ---------------------------------
def normalize(image):
    image = np.abs(image)
    image = image - np.min(image)
    image = image / (np.max(image) + 1e-8)
    return image

img_disp = normalize(img)
dx_disp = normalize(dx)
dy_disp = normalize(dy)
grad_disp = normalize(grad_mag)

# ---------------------------------
# Visualization
# ---------------------------------
plt.figure(figsize=(12, 8))

plt.subplot(2,2,1)
plt.imshow(img_disp, cmap='gray')
plt.title("Original")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(dx_disp, cmap='gray')
plt.title("d/dx (Central Difference)")
plt.axis("off")

plt.subplot(2,2,3)
plt.imshow(dy_disp, cmap='gray')
plt.title("d/dy (Central Difference)")
plt.axis("off")

plt.subplot(2,2,4)
plt.imshow(grad_disp, cmap='gray')
plt.title("|Gradient|")
plt.axis("off")

plt.tight_layout()
plt.show()