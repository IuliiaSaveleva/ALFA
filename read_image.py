import cv2
import PIL.Image as pi
import numpy as np

def read_one_image(image_file):
    try:
        img = cv2.imread(image_file)
        b, g, r = cv2.split(img)
        img_rgb = cv2.merge((r, g, b))
    except:
        print('Error reading image file: ', image_file)
        img_rgb = None
    return img_rgb

def read_images(image_files):
    if isinstance(image_files, str):
        return read_one_image(image_files)
    else:
        images = []
        for image_file in image_files:
            images.append(read_one_image(image_file))
        return images
    return None