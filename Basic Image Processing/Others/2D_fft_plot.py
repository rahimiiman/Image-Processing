import numpy as np
import cv2
import matplotlib.pyplot as plt

# =======================================
# Load Image
# =======================================
img = cv2.imread("test3.jpg", cv2.IMREAD_GRAYSCALE)
img_f = img.astype(np.float32)

# =======================================
# Compute 2D FFT
# =======================================
F = np.fft.fft2(img_f)            # raw FFT
F_shifted = np.fft.fftshift(F)    # shift zero frequency to center

# Magnitude spectra
mag_raw = np.abs(F)
mag_shifted = np.abs(F_shifted)

# Log-magnitude for better visualization
log_mag_raw = np.log1p(mag_raw)
log_mag_shifted = np.log1p(mag_shifted)

# =======================================
# Plot results
# =======================================
plt.figure(figsize=(12,5))

plt.subplot(1,4,1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1,4,2)
plt.imshow(log_mag_raw, cmap='gray')
plt.title("FFT Magnitude (Raw)")
plt.axis('off')

plt.subplot(1,4,3)
plt.imshow(log_mag_shifted, cmap='gray')
plt.title("FFT Magnitude (Shifted)")
plt.axis('off')

plt.subplot(1,4,4)
plt.imshow(log_mag_shifted, cmap='gray')
plt.title("Zoomed Center")
plt.axis('off')
plt.xlim(log_mag_shifted.shape[1]//2-50, log_mag_shifted.shape[1]//2+50)
plt.ylim(log_mag_shifted.shape[0]//2-50, log_mag_shifted.shape[0]//2+50)

plt.tight_layout()
plt.show()