import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load image
image = cv2.imread('test6.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 2. Apply Low Pass Filter (Gaussian Blur)
# ksize (5,5) and sigma 1.0 control the blur "spread"
blurred = cv2.GaussianBlur(image_rgb, (5, 5), 1.0)

# 3. Create the "Mask" (Original - Blurred)
# This represents only the edges/details
mask = cv2.subtract(image_rgb, blurred)

# 4. Add the mask back to the original (Unsharp Masking)
# 'amount' controls how sharp it gets (1.0 is standard)
amount = 3
sharpened = cv2.addWeighted(image_rgb, 1.0 + amount, blurred, -amount, 0)

# 5. Display Comparison
plt.figure(figsize=(15, 5))
titles = ['Original', 'Low Pass (Blurred)', 'Sharpened']
imgs = [image_rgb, blurred, sharpened]

for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(imgs[i])
    plt.title(titles[i])
    plt.axis('off')

plt.show()