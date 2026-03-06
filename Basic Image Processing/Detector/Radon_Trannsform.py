from skimage.transform import radon, iradon
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Parameters
num_of_line_to_plot = 5

# Read grayscale image
img = cv2.imread('test7.jpg', cv2.IMREAD_GRAYSCALE)
rows, cols = img.shape

# Compute Radon transform without circle mask
theta = np.linspace(0., 180., max(img.shape), endpoint=False)
sinogram = radon(img, theta=theta, circle=False)

# Find top peaks in sinogram
peaks = peak_local_max(sinogram, num_peaks=num_of_line_to_plot)

# Create filtered sinogram: only top peaks remain
filtered_sinogram = np.zeros_like(sinogram)
for peak in peaks:
    rho_idx, theta_idx = peak
    filtered_sinogram[rho_idx, theta_idx] = sinogram[rho_idx, theta_idx]

# Apply inverse Radon to get image with only top lines
lines_image = iradon(filtered_sinogram, theta=theta, circle=False)
lines_image = (lines_image / lines_image.max() * 255).astype(np.uint8)

# Resize to match original image in case shapes differ
lines_resized = cv2.resize(lines_image, (cols, rows))
lines_rgb = cv2.cvtColor(lines_resized, cv2.COLOR_GRAY2BGR)

# Overlay red lines on original image
img_with_lines = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
img_with_lines[:, :, 2] = np.clip(img_with_lines[:, :, 2] + lines_rgb[:, :, 0], 0, 255)

# Plot original, sinogram, and result
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Radon Transform\n(Sinogram)')
plt.imshow(sinogram, cmap='gray', aspect='auto')
plt.xlabel('Angle (deg)')
plt.ylabel('Projection position')

plt.subplot(1, 3, 3)
plt.title(f'Top {num_of_line_to_plot} Lines')
plt.imshow(cv2.cvtColor(img_with_lines, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()