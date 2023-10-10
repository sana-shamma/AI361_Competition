import cv2
import numpy as np
import os

# Input and output folders
input_folder = "C:\\Users\\sanas\\Desktop\\AI361Competition\\contest-images"
output_folder = "C:\\Users\\sanas\\Desktop\\AI361Competition\\Week6\\SharpeningAndEdgeEnhancementSharpeningFilter\\output"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load all image files in the input folder
image_files = os.listdir(input_folder)

# Sharpening filter
sharpening_filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

# Process each image
for file_name in image_files:
    # Read the image
    image_path = os.path.join(input_folder, file_name)
    imagedata_original = cv2.imread(image_path)

    # Apply sharpening filter
    sharpened_image = cv2.filter2D(imagedata_original, -1, sharpening_filter)

    # Reduce the size by 75%
    height, width = sharpened_image.shape[:2]
    new_height = int(height * 0.13)
    new_width = int(width * 0.13)
    resized_image = cv2.resize(sharpened_image, (new_width, new_height))

    # Save the result with the same name in the output folder
    output_path = os.path.join(output_folder, file_name)
    cv2.imwrite(output_path, resized_image)