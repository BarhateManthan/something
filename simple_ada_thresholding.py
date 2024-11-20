import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_segmentation_techniques(image_path):

    img = cv2.imread(image_path)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Simple Thresholding
    ret, simple_thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # 2. Adaptive Thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        11, 2
    )
    
    # Display results
    plt.figure(figsize=(12,8))
    
    # Original Image
    plt.subplot(2,2,1)
    plt.title('Original Image')
    plt.imshow(img_rgb)
    plt.axis('off')
    
    # Simple Thresholding
    plt.subplot(2,2,2)
    plt.title('Simple Thresholding')
    plt.imshow(simple_thresh, cmap='gray')
    plt.axis('off')
    
    # Adaptive Thresholding
    plt.subplot(2,2,3)
    plt.title('Adaptive Thresholding')
    plt.imshow(adaptive_thresh, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

    return {
        'simple_threshold': simple_thresh,
        'adaptive_threshold': adaptive_thresh
    }

# Example usage
if __name__ == "__main__":
    # Replace with your image path
    image_path = r'C:\Users\Manthan\Desktop\CV_DL_Practicals\filtering_threshold_otsu_watershed_images_region_growing\train_004.png'
    results = apply_segmentation_techniques(image_path)