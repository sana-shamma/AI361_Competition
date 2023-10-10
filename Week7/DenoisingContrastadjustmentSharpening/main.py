import cv2
import numpy as np

input_image_path = "C:\\Users\\sanas\\Desktop\\AI361Competition\\contest-images\\61.png"
output_image_path = "C:\\Users\\sanas\\Desktop\\AI361Competition\\Week7\\DenoisingContrastadjustmentSharpening\\61.png"

# Read the image
image = cv2.imread(input_image_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Denoise the image
denoised_image = cv2.fastNlMeansDenoising(gray_image, None, 3, 3, 5)

# Enhance contrast
equalized_image = cv2.equalizeHist(denoised_image)

# Apply Gaussian smoothing
smoothed_image = cv2.GaussianBlur(equalized_image, (5, 5), 0)

# Sharpen the image
kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
sharpened_image = cv2.filter2D(smoothed_image, -1, kernel)

# Resize the image by 30% without losing quality
resized_image = cv2.resize(sharpened_image, None, fx=0.7, fy=0.7)

# Save the processed image
cv2.imwrite(output_image_path, resized_image)

print("Image processing and saving completed.")
