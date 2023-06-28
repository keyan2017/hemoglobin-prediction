import cv2
import os
import numpy as np
from PIL import Image

# To achieve the image length and edge equal scale

def img_pad(pil_file):
    # h,w Do not write the order wrong, or the picture will be distorted
    h, w, c = pil_file.shape
    # print(h, w, c)
    fixed_size = 500  # Output the dimensions of the square picture

    if h >= w:
        factor = h / float(fixed_size)
        new_w = int(w / factor)
        if new_w % 2 != 0:
            new_w -= 1
        pil_file = cv2.resize(pil_file, (new_w, fixed_size))
        pad_w = int((fixed_size - new_w) / 2)
        array_file = np.array(pil_file)
        # array_file = np.pad(array_file, ((0, 0), (pad_w, fixed_size-pad_w)), 'constant') #实现黑白图缩放
        array_file = cv2.copyMakeBorder(array_file, 0, 0, pad_w, fixed_size - new_w - pad_w, cv2.BORDER_CONSTANT,
                                        value=(0, 0, 0))  # Black
    else:
        factor = w / float(fixed_size)
        new_h = int(h / factor)
        if new_h % 2 != 0:
            new_h -= 1
        pil_file = cv2.resize(pil_file, (fixed_size, new_h))
        pad_h = int((fixed_size - new_h) / 2)
        array_file = np.array(pil_file)
        # array_file = np.pad(array_file, ((pad_h, fixed_size-pad_h), (0, 0)), 'constant')
        array_file = cv2.copyMakeBorder(array_file, pad_h, fixed_size - new_h - pad_h, 0, 0, cv2.BORDER_CONSTANT,
                                        value=(0, 0, 0))
    output_file = Image.fromarray(array_file)
    return output_file


if __name__ == "__main__":
    dir_image = r'E:\experiment\maskrcnn\testdata\eye_ps'  # Picture folder
    dir_output = r'E:\experiment\maskrcnn\testdata\u_eyeps'  # Output folder
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)
    i = 0
    list_image = os.listdir(dir_image)
    for file in list_image:
        path_image = os.path.join(dir_image, file)
        path_output = os.path.join(dir_output, file)
        pil_image = cv2.imread(path_image)
        b, g, r = cv2.split(pil_image)  # Channel separation and re-merge operation
        pil_image = cv2.merge([r, g, b])
        # print(pil_image)
        # pil_image = pil_image.load()
        output_image = img_pad(pil_image)
        output_image.save(path_output)
        i += 1
        if i % 1000 == 0:
            print('The num of processed images:', i)