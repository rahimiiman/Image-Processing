"""
This code is a template for testing different Structural Elements (SE) in morphological operations. 
You can modify the SE variable to test various shapes and sizes of structural elements.
Four main morphological operations are implemented: Erosion, Dilation, Opening, and Closing.
The results are visualized using Matplotlib, showing the original grayscale image, binary image, 
and the results of each morphological operation along with their differences from the original binary image.
The code consder only morphological operations for binary images.
So to convert a grayscale image to binary, we use absolute thresholding.
You can adjust the threshold value and whether to invert the binary image based on your specific needs and the characteristics of your input images.
"""


import numpy as np
import cv2
import matplotlib.pyplot as plt

# =======================================
# Hyperparameters
# =======================================
input_image_path = "test2.jpg"                                # Path to the input image
threshold = 128                                               # Threshold for binary conversion
invert_binary = False                                         # Whether to invert the binary image (0-->1 , 1--> 0 , useful for certain images)

SE =np.array([[0, 1, 0],
                [1, 1, 1], 
                [0, 1, 0]])  

SE_size = SE.shape[0]  # Assuming SE is square, get the size of the structuring element (e.g., 3 for a 3x3 SE)
# =======================================
# Helper: Binary Conversion
# =======================================
def to_binary(gray_img, threshold=128, invert=False):
    binary = (gray_img >= threshold).astype(np.uint8)
    if invert:
        binary = 1 - binary
    return binary

# =======================================
# Morphological Operations (adaptive SE)
# =======================================
def erosion(image, SE):
    rows, cols = image.shape
    sr, sc = SE.shape
    pad_r, pad_c = sr//2, sc//2                  # integer division to get the padding size (3//2 = 1)
    result = np.zeros_like(image)
    
    for i in range(pad_r, rows - pad_r):
        for j in range(pad_c, cols - pad_c):
            window = image[i-pad_r:i+pad_r+1, j-pad_c:j+pad_c+1]
            if np.all(window[SE == 1] == 1):
                result[i, j] = 1
    return result

def dilation(image, SE):
    # we first need to reflect the SE for dilation, because dilation is the dual of erosion
    SE = np.flip(SE)
    
    rows, cols = image.shape
    sr, sc = SE.shape
    pad_r, pad_c = sr//2, sc//2
    result = np.zeros_like(image)
    
    for i in range(pad_r, rows - pad_r):
        for j in range(pad_c, cols - pad_c):
            window = image[i-pad_r:i+pad_r+1, j-pad_c:j+pad_c+1]
            if np.any(window[SE == 1] == 1):
                result[i, j] = 1
    return result

def opening(image, SE):
    return dilation(erosion(image, SE), SE)

def closing(image, SE):
    return erosion(dilation(image, SE), SE)

# =======================================
# Main
# =======================================
if __name__ == "__main__":

    # Load image
    img = cv2.imread(input_image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binary conversion
    bw = to_binary(gray, threshold=threshold, invert=invert_binary)

    # Apply morphological operations
    morph_erosion = erosion(bw, SE)
    morph_dilation = dilation(bw, SE)
    morph_opening = opening(bw, SE)
    morph_closing = closing(bw, SE)

    # Difference images
    diff_erosion = np.abs(bw - morph_erosion)
    diff_dilation = np.abs(bw - morph_dilation)

    # Plot results
    plt.figure(figsize=(18,6))

    plt.subplot(2,4,1)
    plt.imshow(gray, cmap='gray')
    plt.title("Original Grayscale")
    plt.axis("off")

    plt.subplot(2,4,2)
    plt.imshow(bw, cmap='gray')
    plt.title("Binary Image")
    plt.axis("off")

    plt.subplot(2,4,3)
    plt.imshow(morph_erosion, cmap='gray')
    plt.title(f"Erosion (SE={SE_size}x{SE_size})")
    plt.axis("off")

    plt.subplot(2,4,4)
    plt.imshow(diff_erosion, cmap='gray')
    plt.title("|BW - Erosion|")
    plt.axis("off")

    plt.subplot(2,4,5)
    plt.imshow(morph_dilation, cmap='gray')
    plt.title(f"Dilation (SE={SE_size}x{SE_size})")
    plt.axis("off")

    plt.subplot(2,4,6)
    plt.imshow(diff_dilation, cmap='gray')
    plt.title("|BW - Dilation|")
    plt.axis("off")

    plt.subplot(2,4,7)
    plt.imshow(morph_opening, cmap='gray')
    plt.title(f"Opening (SE={SE_size}x{SE_size})")
    plt.axis("off")

    plt.subplot(2,4,8)
    plt.imshow(morph_closing, cmap='gray')
    plt.title(f"Closing (SE={SE_size}x{SE_size})")
    plt.axis("off")

    plt.tight_layout()
    plt.show()