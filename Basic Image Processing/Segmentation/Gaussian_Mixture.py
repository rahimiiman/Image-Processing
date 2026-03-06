import numpy as np
import cv2
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# 1. Load image and flatten it to a list of pixels
img = cv2.imread('test1.jpg', 0) # Load as grayscale
pixels = img.reshape(-1, 1)

# 2. Define the GMM
# n_components=2 because you have Foreground and Background
gmm = GaussianMixture(n_components=2, covariance_type='tied')

# 3. Fit the model (The EM Algorithm runs here)
gmm.fit(pixels)

# 4. Predict which pixel belongs to which Gaussian (0 or 1)
labels = gmm.predict(pixels)

# 5. Reshape labels back into the original image shape
segmented_img = labels.reshape(img.shape)

# 6. Visualize
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(img, cmap='gray')
plt.subplot(1, 2, 2)
plt.title("Segmented (GMM)")
plt.imshow(segmented_img, cmap='jet')
plt.show()

# To see the calculated means and variances:
print(f"Means: {gmm.means_.flatten()}")
print(f"Variances: {gmm.covariances_.flatten()}")