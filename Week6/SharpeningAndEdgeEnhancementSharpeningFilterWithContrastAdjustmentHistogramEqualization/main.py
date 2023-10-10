import cv2
import numpy as np
import os

def histogram_equalization(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization
    equalized = cv2.equalizeHist(gray)
    
    return equalized

# Specify the input and output folders
input_folder = "C:\\Users\\sanas\\Desktop\\AI361Competition\\contest-images"
output_folder = "C:\\Users\\sanas\\Desktop\\AI361Competition\\Week6\\SharpeningAndEdgeEnhancementSharpeningFilterWithContrastAdjustmentHistogramEqualization\\output"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate over the images in the input folder
for filename in os.listdir(input_folder):
    # Read each image
    image_path = os.path.join(input_folder, filename)
    imagedata_original = cv2.imread(image_path)
    
    # Apply sharpening filter
    sharpening_filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened_image = cv2.filter2D(imagedata_original, -1, sharpening_filter)
    
    # Apply histogram equalization to the sharpened image
    equalized_image = histogram_equalization(sharpened_image)
    
    # Calculate the new dimensions for resizing
    height, width = equalized_image.shape[:2]
    new_width = int(width * 0.20)
    new_height = int(height * 0.20)
    
    # Resize the equalized image
    resized_image = cv2.resize(equalized_image, (new_width, new_height))
    
    # Save the resized image with the same name in the output folder
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, resized_image)

print("Equalized images saved successfully.")