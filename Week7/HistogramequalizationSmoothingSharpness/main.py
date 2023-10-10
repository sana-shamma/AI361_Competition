import cv2
import os

# Input and output folders
input_folder = "C:\\Users\\sanas\\Desktop\\AI361Competition\\contest-images"
output_folder = "C:\\Users\\sanas\\Desktop\\AI361Competition\\Week7\\HistogramequalizationSmoothingSharpness\\output"

# Ensure the output folder exists, create it if necessary
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".png"):  # Ensure you process only PNG images (adjust as needed)
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Load the image
        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        # Apply Contrast Enhancement using Histogram Equalization
        enhanced_image = cv2.equalizeHist(image)

        # Apply Gaussian Smoothing for Noise Reduction
        smoothed_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)

        # Apply Unsharp Masking for Sharpness Enhancement
        unsharp_mask = cv2.addWeighted(enhanced_image, 2.5, smoothed_image, -1.5, 0)

        # Combine the processed images
        combined_image = cv2.addWeighted(enhanced_image, 0.5, unsharp_mask, 0.5, 0)

        # Resize the image to half its original size
        combined_image = cv2.resize(combined_image, None, fx=0.2, fy=0.2)

        # Save the processed image with the same filename to the output folder
        cv2.imwrite(output_path, combined_image)

print("Processing complete. Processed images are saved in the output folder.")
