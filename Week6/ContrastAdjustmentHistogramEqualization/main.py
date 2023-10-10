import cv2
import numpy as np
import os

def histogram_equalization(image):
    # Apply histogram equalization
    equalized = cv2.equalizeHist(image)
    
    return equalized

# Folder paths
input_folder = "C:\\Users\\sanas\\Desktop\\AI361Competition\\contest-images"
output_folder = "C:\\Users\\sanas\\Desktop\\AI361Competition\\Week6\\ContrastAdjustmentHistogramEqualization\\output"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through all the files in the input folder
for filename in os.listdir(input_folder):
    # Check if the file is an image
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        # Load the grayscale image
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path, 0)
        
        # Apply histogram equalization
        equalized_image = histogram_equalization(image)
        
        # Reduce the size of the equalized image by 10%
        resized_image = cv2.resize(equalized_image, None, fx=0.20, fy=0.20)
        
        # Save the equalized image with a reduced size
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, resized_image)

print("Equalized images saved successfully.")