import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Load image
img = cv2.imread('test7.jpg',0).astype(np.float32)
rows, cols = img.shape

# 2. FFT and shift
dft = np.fft.fft2(img)
dft_shift = np.fft.fftshift(dft)

# 3. Create Gaussian Low Pass Filter H_LP(f)
def gaussian_lpf(shape, sigma):
    r, c = shape
    x = np.linspace(-c//2, c//2, c)
    y = np.linspace(-r//2, r//2, r)
    x, y = np.meshgrid(x, y)
    return np.exp(-(x**2 + y**2) / (2 * sigma**2))

# Parameters
sigma = 30    # Cutoff frequency
alpha = 0.6   # Attenuation/Boost (Keep < 1.0 to avoid division by zero)

H_lp = gaussian_lpf((rows, cols), sigma)

# 4. Apply your formula: H_sharp = (1 - alpha * H_lp) / (1 - alpha)
# This amplifies high frequencies while keeping low frequencies at gain 1.0
H_sharp = (1 - alpha * H_lp) / (1 - alpha)

# 5. Filter and Inverse FFT
filtered_shift = dft_shift * H_sharp
filtered_ishift = np.fft.ifftshift(filtered_shift)
img_sharpened = np.real(np.fft.ifft2(filtered_ishift))

# 6. Clip and Display
img_sharpened = np.clip(img_sharpened, 0, 255)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1); plt.imshow(img, cmap='gray'); plt.title('Original')
plt.subplot(1, 2, 2); plt.imshow(img_sharpened, cmap='gray'); plt.title(f'Sharpened (alpha={alpha})')
plt.show()