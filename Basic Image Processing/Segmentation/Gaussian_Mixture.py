"""Gaussian Mixture Model for Image Segmentation"""
"""
This code demonstrates how to use Gaussian Mixture Models (GMM) for image segmentation.
inGMM is a probabilistic model that assumes data points are generated from a mixture of several 
Gaussian distributions with unknown parameters.

GMM usually is used for images that their histogram has multiple peaks,
which indicates the presence of multiple distinct clusters.

The only Hyperparameter is the number of clusters (K), which can be determined by 
analyzing the histogram of the image 

So when working with highly dynamic images this known hyperparameter can be a problem, 
but for static images it can be a good choice for segmentation."""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

#=========================
# Hyperparameters
#=========================
K = 5              # Number of clusters (Gaussian components)
image_path = "test4.jpg"  # Path to input image
is_gray = False               # Whether to convert image to grayscale for GMM
#=========================
# Load Image and analysis histogram
#=========================
if is_gray:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32)
    X = img.reshape(-1, 1) # Reshape to pixel list (N, 1)
    c=1
    h, w = img.shape
else:
    img = cv2.imread(image_path)
    h, w, c = img.shape

    if c==3:
        print("Image has 3 channels (RGB)")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    X = img.reshape(-1, c) # Reshape image to pixel list

# Colors for plotting
colors = ['r', 'g', 'b']

# Create one figure with subplots
plt.figure(figsize=(15,4))

for i in range(c):
    plt.subplot(1, c, i+1)  # 1 row, c columns, i+1-th subplot
    plt.hist(X[:, i], bins=256, color=colors[i], alpha=0.7)
    plt.title(f'Channel {i}')
    plt.xlabel("Pixel value")
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# Fit GMM
gmm = GaussianMixture(n_components=K, covariance_type='tied', random_state=0)
gmm.fit(X)

# Predict cluster labels
labels = gmm.predict(X)

# Reshape labels to image
segmented_labels = labels.reshape(h, w)

# Reconstruct segmented image using cluster means
segmented_img = gmm.means_[labels]
segmented_img = segmented_img.reshape(h, w, c).astype(np.uint8)

# ========================
# Plot results
# ========================

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Original Image")
if is_gray:
    plt.imshow(img, cmap='gray')
else:
    plt.imshow(img)
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Cluster Labels")
plt.imshow(segmented_labels, cmap='viridis')
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Segmented Image")
if is_gray:
    plt.imshow(segmented_img, cmap='gray')  
else:   
    plt.imshow(segmented_img)
plt.axis("off")

plt.tight_layout()
plt.show()