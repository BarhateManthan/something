import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_filters(image_path):

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
    
    median_filtered = cv2.medianBlur(img, 5)
    
    gaussian_filtered = cv2.GaussianBlur(img, (5,5), 0)
    
    plt.figure(figsize=(12,8))
    
    # Original Image
    plt.subplot(2,2,1)
    plt.title('Original Image')
    plt.imshow(img)
    plt.axis('off')
    
    # Sobel Filter
    plt.subplot(2,2,2)
    plt.title('Sobel Filter (Edge Detection)')
    plt.imshow(sobel_combined, cmap='gray')
    plt.axis('off')
    
    # Median Filter
    plt.subplot(2,2,3)
    plt.title('Median Filter (Noise Reduction)')
    plt.imshow(median_filtered)
    plt.axis('off')
    
    # Gaussian Filter
    plt.subplot(2,2,4)
    plt.title('Gaussian Filter (Smoothing)')
    plt.imshow(gaussian_filtered)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Call the function 
apply_filters(r'C:\Users\Manthan\Desktop\CV_DL_Practicals\filtering_threshold_otsu_watershed_images_region_growing\train_019.png')