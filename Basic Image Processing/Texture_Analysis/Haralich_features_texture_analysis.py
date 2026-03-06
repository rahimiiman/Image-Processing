import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops

# -------------------------------------------------
# 1. LOAD IMAGE (CHANGE PATH)
# -------------------------------------------------
image_path = r"test6.jpg"   # <-- CHANGE THIS
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise ValueError("Image not found. Check path.")

# -------------------------------------------------
# 2. PARAMETERS
# -------------------------------------------------
window_size = 11     # sliding window size
stride = 10          # step size
levels = 32          # quantized gray levels
distances = [1]
angles = [0]

properties = [
    'contrast',
    'dissimilarity',
    'homogeneity',
    'energy',
    'correlation',
    'ASM'
]

# -------------------------------------------------
# 3. QUANTIZE IMAGE (0 → levels-1)
# -------------------------------------------------
image_q = (image / (256 / levels)).astype(np.uint8)

# -------------------------------------------------
# 4. PREPARE FEATURE MAPS
# -------------------------------------------------
h, w = image_q.shape
out_h = (h - window_size) // stride + 1
out_w = (w - window_size) // stride + 1

feature_maps = {prop: np.zeros((out_h, out_w)) for prop in properties}

# -------------------------------------------------
# 5. SLIDING WINDOW
# -------------------------------------------------
for y in range(0, h - window_size + 1, stride):
    for x in range(0, w - window_size + 1, stride):

        window = image_q[y:y+window_size, x:x+window_size]

        glcm = graycomatrix(window,
                            distances=distances,
                            angles=angles,
                            levels=levels,
                            symmetric=True,
                            normed=True)

        i = y // stride
        j = x // stride

        for prop in properties:
            value = graycoprops(glcm, prop)[0, 0]
            feature_maps[prop][i, j] = value

# -------------------------------------------------
# 6. PLOT RESULTS
# -------------------------------------------------
plt.figure(figsize=(12, 8))

plt.subplot(2, 4, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

k = 2
for prop in properties:
    plt.subplot(2, 4, k)
    plt.imshow(feature_maps[prop], cmap='jet')
    plt.title(prop)
    plt.colorbar()
    plt.axis('off')
    k += 1

plt.tight_layout()
plt.show()