import numpy as np
import cv2
import matplotlib.pyplot as plt

# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------
cutoff = 5           # Cutoff frequency for low-pass filter
c = 1                 # Sharpness of the filter transition
mu0 = 0.5            # Desired global mean after normalization
sigma0 = 0.05         # Desired global standard deviation after normalization
input_image_path = 'test5.jpg'  # Path to input image

# -------------------------------------------------
# Gaussian Low-Pass Filter
# -------------------------------------------------
def low_pass_filter(image, cutoff, c=1):
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    dist_sq = x * x + y * y

    # Gaussian low-pass
    L = np.exp(-c * dist_sq / (cutoff ** 2))

    return L

# -------------------------------------------------
# Second-Degree Homomorphic Normalization
# -------------------------------------------------
def second_degree_homomorphic_channel(img_channel, cutoff=20, c=1, mu0=0.9, sigma0=0.02, epsilon=1e-8):
    # Normalize to [0,1]
    img = img_channel.astype(np.float32) / 255.0

    # Fourier transform
    G = np.fft.fftshift(np.fft.fft2(img))

    # Low-pass filter
    L = low_pass_filter(img, cutoff, c)

    # Local mean estimation (high-pass)
    g_centered = np.real(np.fft.ifft2(np.fft.ifftshift(G * (1 - L))))

    # Local variance estimation
    g2 = g_centered ** 2
    G2 = np.fft.fftshift(np.fft.fft2(g2))
    var_hat = np.real(np.fft.ifft2(np.fft.ifftshift(G2 * L)))
    sigma_hat = np.sqrt(var_hat + epsilon)

    # Normalize
    s_hat = g_centered / (sigma_hat + epsilon)
    gamma = sigma0 * s_hat + mu0
    gamma = np.clip(gamma, 0, 1)

    return (gamma * 255).astype(np.uint8)

# -------------------------------------------------
# Wrapper for grayscale or color
# -------------------------------------------------
def second_degree_homomorphic(image, cutoff=20, c=1, mu0=0.9, sigma0=0.02):
    if len(image.shape) == 2:
        # Grayscale image
        return second_degree_homomorphic_channel(image, cutoff, c, mu0, sigma0)
    elif len(image.shape) == 3:
        # Color image: convert to HSV, enhance V channel
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        v_enhanced = second_degree_homomorphic_channel(v, cutoff, c, mu0, sigma0)

        hsv_enhanced = cv2.merge([h, s, v_enhanced])
        rgb_enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2RGB)
        return rgb_enhanced
    else:
        raise ValueError("Unsupported image format")

# -------------------------------------------------
# Example Usage
# -------------------------------------------------
if __name__ == "__main__":

    img = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = second_degree_homomorphic(
        img_rgb,
        cutoff=cutoff,
        c=c,
        mu0=mu0,
        sigma0=sigma0
    )

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(result)
    plt.title("Second-Degree Homomorphic")
    plt.axis("off")

    plt.tight_layout()
    plt.show()