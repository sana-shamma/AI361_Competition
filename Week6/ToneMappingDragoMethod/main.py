import cv2
import numpy as np

def readImagesAndTimes():
    filenames = ["0.png"]
    images = []
    for filename in filenames:
        im = cv2.imread(filename)
        images.append(im)
    return images

if __name__ == '__main__':
    # Read images
    print("Reading images ...")
    images = readImagesAndTimes()

    # Tone mapping using Drago method
    print("Applying tone mapping using Drago method ...")
    tonemapped_images = []
    tonemapper = cv2.createTonemapDrago()

    for image in images:
        # Convert image to float32 format and normalize pixel values
        image_float = image.astype(np.float32) / 255.0

        # Apply tonemapping
        tonemapped = tonemapper.process(image_float)

        # Convert tonemapped image back to uint8 format
        tonemapped_uint8 = (tonemapped * 255).astype(np.uint8)
        tonemapped_images.append(tonemapped_uint8)

    # Resize the tonemapped images to original size
    original_size_images = []
    for image in tonemapped_images:
        resized_image = cv2.resize(image, (image.shape[1]*2, image.shape[0]*2))
        original_size_images.append(resized_image)

    # Save the tonemapped images in original size
    for i, tonemapped in enumerate(original_size_images):
        filename = f"tonemapped_image_{i}.png"
        cv2.imwrite(filename, tonemapped)

    print("Tonemapped images saved.")