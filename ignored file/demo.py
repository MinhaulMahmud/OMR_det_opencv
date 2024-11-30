import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the image
image_path = 'unnamed (1).jpg'  # Replace with the actual file path
image = cv2.imread(image_path)

# Step 2: Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 3: Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Step 4: Apply Adaptive Thresholding to highlight regions
thresholded = cv2.adaptiveThreshold(
    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
)

# Step 5: Apply Morphological Operations
kernel = np.ones((5, 5), np.uint8)
morph = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
morph = cv2.dilate(morph, kernel, iterations=2)
morph = cv2.erode(morph, kernel, iterations=1)

# Step 6: Detect edges using Canny edge detection
edges = cv2.Canny(morph, 50, 150)

# Step 7: Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 8: Find the largest contours and filter for rectangular shapes
contours = sorted(contours, key=cv2.contourArea, reverse=True)
omr_contour = None

for contour in contours:
    # Approximate the contour to a polygon
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # Check if the contour is a rectangle (4 points) and large enough
    if len(approx) == 4 and cv2.contourArea(approx) > 10000:
        omr_contour = approx
        break

# Step 9: Draw the detected OMR contour on the original image
result_image = image.copy()
if omr_contour is not None:
    cv2.drawContours(result_image, [omr_contour], -1, (0, 255, 0), 3)  # Green contour

# Step 10: Apply perspective transformation if a rectangle is detected
if omr_contour is not None:
    # Get the bounding box coordinates
    omr_contour = omr_contour.reshape(4, 2)
    
    # Define the destination points for perspective transformation
    (tl, tr, br, bl) = omr_contour
    width = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
    height = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))
    dst_points = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")
    
    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(omr_contour.astype("float32"), dst_points)
    
    # Apply the perspective transformation
    aligned_image = cv2.warpPerspective(image, M, (int(width), int(height)))
else:
    aligned_image = result_image

# Display all intermediate steps and the final result
plt.figure(figsize=(18, 12))

# Original Grayscale Image
plt.subplot(2, 3, 1)
plt.title("Grayscale Image")
plt.imshow(gray, cmap="gray")
plt.axis("off")

# Blurred Image
plt.subplot(2, 3, 2)
plt.title("Blurred Image")
plt.imshow(blurred, cmap="gray")
plt.axis("off")

# Thresholded Image
plt.subplot(2, 3, 3)
plt.title("Thresholded Image")
plt.imshow(thresholded, cmap="gray")
plt.axis("off")

# Morphological Operations
plt.subplot(2, 3, 4)
plt.title("Morphological Operations")
plt.imshow(morph, cmap="gray")
plt.axis("off")

# Edge-Detected Image
plt.subplot(2, 3, 5)
plt.title("Edge-Detected Image")
plt.imshow(edges, cmap="gray")
plt.axis("off")

# Detected OMR Contour
plt.subplot(2, 3, 6)
plt.title("Detected OMR Contour")
plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
plt.axis("off")

# Aligned OMR Sheet
plt.subplot(2, 3, 7)
plt.title("Aligned OMR Sheet")
plt.imshow(cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.tight_layout()
plt.show()
