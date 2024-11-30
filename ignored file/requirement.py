import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "unnamed (1).jpg"  # Replace with your actual image path
image = cv2.imread(image_path)

# Resize the image for easier processing
height, width = image.shape[:2]
resize_factor = 1200 / width  # Scale down the width to 1200 pixels
image_resized = cv2.resize(image, (int(width * resize_factor), int(height * resize_factor)))

# Convert the image to grayscale
gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Detect edges using Canny edge detection
edges = cv2.Canny(blurred, 50, 150)

# Find contours in the edge-detected image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort the contours by area (largest first)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# Filter out small contours that are unlikely to be the answer sections
answer_sections = []

# We already have the first answer section, let's assume it's the first contour
first_section_contour = contours[0]

# Approximate the first section contour to find its four corners
epsilon = 0.02 * cv2.arcLength(first_section_contour, True)
approx = cv2.approxPolyDP(first_section_contour, epsilon, True)

# Check if we detected exactly four corners for the first section
if len(approx) == 4:
    print("Detected the first answer section!")

    # Add the first detected section to the list
    answer_sections.append(approx)

# Now, we need to find the remaining three answer sections
# Iterate through the remaining contours
for contour in contours[1:]:
    # Approximate the contour to find the four corners
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Check if the contour has four corners (likely an answer section)
    if len(approx) == 4:
        # Check the area of the detected section to make sure it's similar to the first one
        if cv2.contourArea(contour) > 1000:  # Minimum area threshold to exclude small contours
            answer_sections.append(approx)

    # Stop if we've found 4 sections
    if len(answer_sections) >= 4:
        break

# Display the detected answer sections
output_image = image_resized.copy()

# Draw all detected answer sections on the image
for section in answer_sections:
    cv2.drawContours(output_image, [section], -1, (0, 255, 0), 3)

# Display the result
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.title("Detected Answer Sections")
plt.axis('off')
plt.show()

# Check how many sections we detected
if len(answer_sections) < 4:
    print(f"Warning: Only {len(answer_sections)} answer sections detected.")
else:
    print("Successfully detected 4 answer sections!")
