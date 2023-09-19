import os
import cv2
from PIL import Image
from rembg import remove


def process_image(input_path, output_path):
    # Processing the image
    input = Image.open(input_path)

    # Removing the background from the given image
    output = remove(input)
    os.remove(input_path)

    # Saving the image in the given path
    output.save(output_path)

    png_img = cv2.imread(output_path)
    os.remove(output_path)

    # Save the image back in its original input path
    cv2.imwrite(input_path, png_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])