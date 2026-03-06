"""
This code is used to find the location of basic pattern in structural texture. 
The idea is that , if we can separate the principle pattern in frequency domain (around (0,0) in frequency domain)
Inverst Fourier transform is an image that gives us a coarse estimation of pattern.

As we only need the principle frequency replica , cutoff frequency of lowpass filter must beselected in a proper way
The filtering is done in frequency domain by multiplication and then reverted back to spatial domain by inverse Fourier transform.
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt

## Hyperparameters

image_path = r"test4.jpg"   # <-- CHANGE THIS
filter_cutoff = 5   # D0

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    raise ValueError("Image not found. Check the file path.")
image = image.astype(np.float32)


# =========================
#  Compute DFT
# =========================
dft = np.fft.fft2(image)
dft_shift = np.fft.fftshift(dft)

rows, cols = image.shape
crow, ccol = rows // 2, cols // 2


# =========================
#  Filter
# =========================
cutoff = filter_cutoff   

y, x = np.ogrid[:rows, :cols]
D2 = (y - crow)**2 + (x - ccol)**2         ## this is distance from the center of frequency domain

H = np.exp(-D2 / (2 * (cutoff**2)))

# =========================
# Apply Filter
# =========================
filtered_dft = dft_shift * H


# =========================
# 6️⃣ Compute IDFT
# =========================
filtered_ishift = np.fft.ifftshift(filtered_dft)
image_idft = np.fft.ifft2(filtered_ishift)
image_idft = np.real(image_idft)

# Normalize to 0–255
image_exp = np.clip(image_idft, 0, 255)
image_exp = image_exp.astype(np.uint8)


# =========================
# Plot Results
# =========================
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(image.astype(np.uint8), cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title("Homomorphic Filtered Image")
plt.imshow(image_exp, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()