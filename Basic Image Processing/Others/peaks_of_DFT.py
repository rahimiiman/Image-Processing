import numpy as np
import cv2
import matplotlib.pyplot as plt

# =====================================
# Load image
# =====================================
img = cv2.imread("test4.jpg", cv2.IMREAD_GRAYSCALE)
img_f = img.astype(np.float32)

# =====================================
# Compute DFT
# =====================================
F = np.fft.fft2(img_f)
F_shifted = np.fft.fftshift(F)

magnitude = np.abs(F_shifted)
log_mag = np.log1p(magnitude)

rows, cols = magnitude.shape
crow, ccol = rows // 2, cols // 2

# =====================================
# Detect Peaks
# =====================================
flat_idx = np.argsort(magnitude.ravel())[::-1]

peaks = []
for idx in flat_idx:
    r, c = np.unravel_index(idx, magnitude.shape)

    # Skip center initially
    if (r == crow and c == ccol):
        continue

    # Avoid very close to center
    if np.sqrt((r-crow)**2 + (c-ccol)**2) < 2:
        continue

    peaks.append((r, c))

    if len(peaks) == 4:
        break

# Center
center = (crow, ccol)

# =====================================
# Compute radius using first peak
# =====================================
r1, c1 = peaks[0]
dist = np.sqrt((r1-crow)**2 + (c1-ccol)**2)
radius = int(1.5 * dist)

# =====================================
# Create mask
# =====================================
mask = np.zeros_like(F_shifted, dtype=np.float32)

Y, X = np.ogrid[:rows, :cols]

# Keep DC region
center_mask = (X-ccol)**2 + (Y-crow)**2 <= radius**2
mask[center_mask] = 1

# Keep 4 peak regions
for (rp, cp) in peaks:
    peak_mask = (X-cp)**2 + (Y-rp)**2 <= radius**2
    mask[peak_mask] = 1

# =====================================
# Apply mask
# =====================================
F_separated = F_shifted * mask

# Inverse FFT
F_ishift = np.fft.ifftshift(F_separated)
reconstructed = np.real(np.fft.ifft2(F_ishift))

# =====================================
# Plot results
# =====================================
plt.figure(figsize=(14,8))

# 1️⃣ Original
plt.subplot(2,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis("off")

# 2️⃣ Full FFT + Peaks
plt.subplot(2,2,2)
plt.imshow(log_mag, cmap='gray')
plt.title("Full FFT (Log)")
plt.axis("off")

# Mark center
plt.plot(ccol, crow, 'r*', markersize=12)

# Mark peaks
for (r, c) in peaks:
    plt.plot(c, r, 'r*', markersize=12)

# 3️⃣ Zoomed Masked FFT
plt.subplot(2,2,3)
plt.imshow(np.log1p(np.abs(F_separated)), cmap='gray')
plt.title("Separated Frequency Region")
plt.axis("off")

# 4️⃣ Reconstructed Image
plt.subplot(2,2,4)
plt.imshow(reconstructed, cmap='gray')
plt.title("Inverse FFT of Separated Area")
plt.axis("off")

plt.tight_layout()
plt.show()