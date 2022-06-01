import os
import cv2
import errno

from image_eraser import erase

SOURCE_FOLDER = "images/cat/"
OUT_FOLDER = "images/cat/"


if __name__ == "__main__":
    output_dir = os.path.join(OUT_FOLDER)
    
    try:
        os.makedirs(output_dir)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    
    image_name = SOURCE_FOLDER + 'input_1.jpg'
    mask_name = SOURCE_FOLDER + 'mask_1.jpg'

    image = cv2.imread(image_name)
    mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)  # one channel only

    output = erase(image, mask, window=(22,22))
    cv2.imwrite(OUT_FOLDER + 'result_1.png', output)
