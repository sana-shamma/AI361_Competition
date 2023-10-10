
import cv2
import os
import numpy as np

# Specify the folder containing the input HDR images
input_folder = "C:\\Users\\sanas\\Desktop\\AI361Competition\\contest-images"

# Specify the folder to save the output LDR images
output_folder = "C:\\Users\\sanas\\Desktop\\AI361Competition\\Week7\\DenoisingContrastadjustmentSharpeningReinhard\\output"

# Tonemapping parameters
gamma = 0.5
contrast = 0
saturation = 0
detail = 0

# Get a list of all files in the input folder
file_list = os.listdir(input_folder)

# Process each file in the input folder
for filename in file_list:
    # Check if the file is an image
    if filename.endswith((".jpg", ".jpeg", ".png")):
        # Load the HDR image
        hdr_path = os.path.join(input_folder, filename)
        hdr_image = cv2.imread(hdr_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)

        # Convert the HDR image to the correct format
        hdr_image = hdr_image.astype('float32') / 65535.0  # Normalize to the range [0, 1]

        # Tonemap using Reinhard's method
        tonemap_reinhard = cv2.createTonemapReinhard(gamma, contrast, saturation, detail)
        ldr_reinhard = tonemap_reinhard.process(hdr_image)

        # Convert the LDR image to the correct format for further processing
        ldr_image = (ldr_reinhard * 255).astype('uint8')

        # Perform image processing operations
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(ldr_image, cv2.COLOR_BGR2GRAY)

        # Denoise the image
        denoised_image = cv2.fastNlMeansDenoising(gray_image, None, 10, 10, 7)

        # Enhance contrast
        equalized_image = cv2.equalizeHist(denoised_image)

        # Apply Gaussian smoothing
        smoothed_image = cv2.GaussianBlur(equalized_image, (5, 5), 0)

        # Sharpen the image
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened_image = cv2.filter2D(smoothed_image, -1, kernel)

        # Resize the image by 30% without losing quality
        resized_image = cv2.resize(sharpened_image, None, fx=0.2, fy=0.2)

        # Save the processed image
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, resized_image)

        print(f"Saved {output_path}")

print("Tonemapping and image processing completed.")