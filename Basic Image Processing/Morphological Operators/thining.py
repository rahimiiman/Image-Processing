import cv2
import numpy as np
import matplotlib.pyplot as plt


def thinning(binary_img):
    """
    Perform morphological thinning using 8 structuring elements.
    Input must be binary image with values {0,1}.
    """

    # Structuring elements (int8 required)
    S = [
        np.array([[-1, -1, -1], [ 0,  1,  0], [ 1,  1,  1]], dtype=np.int8),
        np.array([[ -1, -1,  0], [-1,  1, 1], [ 0,  1,  1]], dtype=np.int8),
        np.array([[ -1,  0,  1], [ -1,  1, 1], [ -1,  0, 1]], dtype=np.int8),
        np.array([[ 0,  1,  1], [ -1,  1, 1], [ -1, -1,  0]], dtype=np.int8),
        np.array([[ 1,  1,  1], [ 0,  1,  0], [-1, -1, -1]], dtype=np.int8),
        np.array([[ 1,  1,  0], [1,  1,  -1], [ 0, -1,  -1]], dtype=np.int8),
        np.array([[1,  0,  -1], [1,  1,  -1], [1,  0,  -1]], dtype=np.int8),
        np.array([[ 0, -1,  -1], [1, 1,  -1], [ 1,  1,  0]], dtype=np.int8)
    ]

    img = binary_img.copy().astype(np.uint8)

    changed = True
    while changed:
        changed = False
        for kernel in S:
            hitmiss = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel)
            if np.any(hitmiss):
                img = cv2.subtract(img, hitmiss)
                changed = True

    return img


if __name__ == "__main__":

    # Read grayscale
    img = cv2.imread("test1.jpg", cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Error loading image.")
        exit()

    # Otsu threshold
    _, binary = cv2.threshold(img, 0, 1,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ensure foreground = 1
    binary = binary.astype(np.uint8)

    # Apply thinning
    thinned = thinning(binary)

    # Convert to 0-255 for visualization
    binary_display = binary * 255
    thinned_display = thinned * 255

    # Show results
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Binary Image")
    plt.imshow(binary_display, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Thinned Image")
    plt.imshow(thinned_display, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()