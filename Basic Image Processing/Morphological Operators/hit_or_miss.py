import numpy as np
import cv2
import matplotlib.pyplot as plt

# =======================================
# Helper: Binary Conversion
# =======================================
def to_binary(gray_img, threshold=128, invert=False):
    """
    Convert grayscale image to binary (0 and 1).
    """
    binary = (gray_img >= threshold).astype(np.uint8)
    if invert:
        binary = 1 - binary
    return binary

# =======================================
# Hit-or-Miss Operator
# =======================================
def hit_or_miss(image, SE):
    """
    Perform Hit-or-Miss on a binary image with a ternary SE.
    
    SE values:
        1  -> must be foreground (hit)
       -1  -> must be background (miss)
        0  -> don't care
    """
    rows, cols = image.shape
    sr, sc = SE.shape
    pad_r, pad_c = sr//2, sc//2
    result = np.zeros_like(image)
    
    for i in range(pad_r, rows - pad_r):
        for j in range(pad_c, cols - pad_c):
            window = image[i-pad_r:i+pad_r+1, j-pad_c:j+pad_c+1]
            
            hit_mask = (SE == 1)
            miss_mask = (SE == -1)
            
            hit_check = np.all(window[hit_mask] == 1) if np.any(hit_mask) else True
            miss_check = np.all(window[miss_mask] == 0) if np.any(miss_mask) else True
            
            if hit_check and miss_check:
                result[i, j] = 1
    
    return result

# =======================================
# Main
# =======================================
if __name__ == "__main__":
    # Load image
    img = cv2.imread("test1.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binary conversion
    bw = to_binary(gray, threshold=128, invert=False)

    # Define Hit-or-Miss Structuring Element
    # Example: cross-shaped hit with background at corners
    SE = np.array([[ 0,  -1, -1],
                   [  1,  1,  -1],
                   [ 1,  1, 0]], dtype=np.int8)

    # Apply Hit-or-Miss
    hom_result = hit_or_miss(bw, SE)

    # Difference image (optional)
    diff = np.abs(bw - hom_result)

    # Plot results
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(gray, cmap='gray')
    plt.title("Original Grayscale")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(bw, cmap='gray')
    plt.title("Binary Image")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(hom_result, cmap='gray')
    plt.title("Hit-or-Miss Result")
    plt.axis("off")

    plt.tight_layout()
    plt.show()