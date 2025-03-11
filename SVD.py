import numpy as np
import cv2
import matplotlib.pyplot as plt

def compress_grayscale(image, k):
    U, S, Vt = np.linalg.svd(image, full_matrices=False)
    compressed = (U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]).astype(np.uint8)
    return compressed

def compress_color(image, k):
    compressed_channels = []
    for i in range(3):  # Iterate through BGR channels
        U, S, Vt = np.linalg.svd(image[:, :, i], full_matrices=False)
        compressed_channel = (U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]).astype(np.uint8)
        compressed_channels.append(compressed_channel)
    return cv2.merge(compressed_channels)

def main():
    image_path = input("Enter the path of the image: ")
    k = int(input("Enter the number of singular values to retain: "))
    
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return
    
    if len(image.shape) == 2:  # Grayscale image
        compressed_image = compress_grayscale(image, k)
    else:  # Colored image
        compressed_image = compress_color(image, k)
    
    # Display original and compressed images
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    if len(image.shape) == 2:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title(f"Compressed Image (k={k})")
    if len(image.shape) == 2:
        plt.imshow(compressed_image, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(compressed_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.show()
    
    # Save the compressed image
    output_path = "compressed_image.jpg"
    cv2.imwrite(output_path, compressed_image)
    print(f"Compressed image saved as {output_path}")

if __name__ == "__main__":
    main()