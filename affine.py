import cv2
import numpy as np
import matplotlib.pyplot as plt

def geometric_transformations(image_path):
    # Read the image
    img = cv2.imread(image_path)
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # 1. Scaling
    scale_factor = 0.5
    scaled = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)
    
    # 2. Translation
    tx, ty = 50, 30  # translate by 50 pixels right and 30 pixels down
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    translated = cv2.warpAffine(img, translation_matrix, (width, height))
    
    # 3. Rotation
    angle = 45  # rotate by 45 degrees
    rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1.0)
    rotated = cv2.warpAffine(img, rotation_matrix, (width, height))
    
    # 4. Shearing
    shear_matrix = np.float32([[1, 0.5, 0], [0, 1, 0]])
    sheared = cv2.warpAffine(img, shear_matrix, (width, height))
    
    # 5. Reflection (Flipping)
    # Horizontal flip
    horizontal_flip = cv2.flip(img, 1)
    # Vertical flip
    vertical_flip = cv2.flip(img, 0)
    
    # Create subplot layout
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    fig.suptitle('Geometric Transformations', fontsize=16)
    
    # Plot all transformations
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original')
    
    axes[0, 1].imshow(scaled)
    axes[0, 1].set_title('Scaled (0.5x)')
    
    axes[0, 2].imshow(translated)
    axes[0, 2].set_title('Translated')
    
    axes[0, 3].imshow(rotated)
    axes[0, 3].set_title('Rotated (45°)')
    
    axes[1, 0].imshow(sheared)
    axes[1, 0].set_title('Sheared')
    
    axes[1, 1].imshow(horizontal_flip)
    axes[1, 1].set_title('Horizontal Flip')
    
    axes[1, 2].imshow(vertical_flip)
    axes[1, 2].set_title('Vertical Flip')
    
    # Remove extra subplot
    axes[1, 3].remove()
    
    # Remove axes ticks
    for ax in axes.flat:
        if ax._position:  # Check if subplot exists
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()


# Usage
if __name__ == "__main__":
    image_path = r"C:\Users\Manthan\Desktop\CV_DL_Practicals\jpg_affine_images\4.jpg"
    geometric_transformations(image_path)