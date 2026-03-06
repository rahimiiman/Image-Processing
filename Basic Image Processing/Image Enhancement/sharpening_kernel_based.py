import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the image
# OpenCV loads images in BGR format by default
image_bgr = cv2.imread('test3.jpg')

# 2. Sharpening Logic (Kernel Method)
kernel = np.array([[-0.005, -0.005, -0.005], 
                   [-0.005, 1.4, -0.005], 
                   [-0.005, -0.005, -0.005]])
sharpened_bgr = cv2.filter2D(image_bgr, -1, kernel)

# 3. Convert BGR to RGB for Matplotlib display
# Matplotlib expects RGB, otherwise colors will look "swapped"
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
sharpened_rgb = cv2.cvtColor(sharpened_bgr, cv2.COLOR_BGR2RGB)

# 4. Create the side-by-side comparison plot
plt.figure(figsize=(12, 6))

# Subplot 1: Original
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off') # Hide pixel coordinates

# Subplot 2: Sharpened
plt.subplot(1, 2, 2)
plt.imshow(sharpened_rgb)
plt.title('Sharpened Image')
plt.axis('off')

# Display the final comparison
plt.tight_layout()
plt.show()