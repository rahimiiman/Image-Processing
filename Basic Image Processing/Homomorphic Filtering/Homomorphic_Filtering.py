import cv2
import numpy as np
import matplotlib.pyplot as plt

###Hyperparameters
low_gain = 0.2        # Low-frequency gain (low frequency components will be attenuated by this factor)
high_gain = 1.5       # High-frequency gain (high frequency components will be amplified by this factor)
cutoff = 20           # Cutoff frequency for the filter (in pixels, determines the radius of the low-pass region)
c = 1                 # Sharpness of the filter transition (controls how quickly the filter transitions from low_gain to high_gain)
input_image_path = 'test3.jpg'  # Path to input image

def homomorphic_filter(image, low_gain=0.4, high_gain=1.8, cutoff=40, c=1):
    image = image.astype(np.float32)
    image = image / 255.0 + 1e-6

    img_log = np.log(image)

    dft = np.fft.fft2(img_log)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    dist_sq = x*x + y*y

    h_uv = high_gain - (high_gain - low_gain) * np.exp(-c * dist_sq / (cutoff**2))

    # Explicitly set the DC component (center of the shifted FFT) to 1
    h_uv[crow, ccol] = 1

    filtered = dft_shift * h_uv

    img_back = np.fft.ifft2(np.fft.ifftshift(filtered))
    img_back = np.real(img_back)

    img_exp = np.exp(img_back)
    img_exp = np.clip(img_exp, 0, 1)

    result = (img_exp * 255).astype(np.uint8)

    return result


# ---------------------------
# Read image (color)
# ---------------------------
img = cv2.imread(input_image_path)

if img is None:
    raise FileNotFoundError("Error: test1.jpg not found.")

# Check if grayscale
if len(img.shape) == 2:
    # Grayscale image
    enhanced_img = homomorphic_filter(img, low_gain, high_gain, cutoff, c)
    display_original = img
    display_enhanced = enhanced_img

else:
    # Color image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Apply homomorphic filtering only to V channel
    v_filtered = homomorphic_filter(v, low_gain, high_gain, cutoff, c)

    # Merge back
    hsv_filtered = cv2.merge([h, s, v_filtered])
    enhanced_bgr = cv2.cvtColor(hsv_filtered, cv2.COLOR_HSV2BGR)
    enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

    display_original = img_rgb
    display_enhanced = enhanced_rgb


# ---------------------------
# Show images side by side
# ---------------------------
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.imshow(display_original)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(display_enhanced)
plt.title("Homomorphic Enhanced")
plt.axis("off")

plt.tight_layout()
plt.show()