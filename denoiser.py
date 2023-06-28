import cv2
import os
import numpy as np
import utils_paths
from sklearn.cluster import KMeans
from PIL import Image

# Read image address
data_path = r'F:/experiment/blood/neweye'
imagePaths = sorted(list(utils_paths.list_images(data_path)))
path_output = 'F:/experiment/blood/out_put'

# means clustering images

# Display picture function
def show(winname,src):
    cv2.namedWindow(winname,cv2.WINDOW_GUI_NORMAL)
    cv2.imshow(winname,src)
    cv2.waitKey()

# Load color images through a loop
for imagePath in imagePaths:
    img = cv2.imread(imagePath)


    # 分离BGR通道，并转换为灰度值
    b, g, r = cv2.split(img)
    gray_b = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
    gray_g = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    gray_r = cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)
    gray_b_BB = gray_b[:, :, 0].reshape(-1,1)
    gray_g_GG = gray_g[:, :, 0].reshape(-1,1)
    gray_r_RR = gray_r[:, :, 0].reshape(-1,1)


    # The rgb value of a pixel is treated as a unit
    data_b = gray_b.reshape((-1,3))
    data_g = gray_g.reshape((-1,3))
    data_r = gray_r.reshape((-1,3))

    # Convert data type
    data_b = np.float32(data_b)
    data_g = np.float32(data_g)
    data_r = np.float32(data_r)

    # Set the Kmeans parameter
    critera = (cv2.TermCriteria_EPS+cv2.TermCriteria_MAX_ITER,10,0.1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    # Classify the pictures into three categories
    bb, best_b, center_b = cv2.kmeans(data_b, 3, None, criteria=critera, attempts=10, flags=flags)
    gg, best_g, center_g = cv2.kmeans(data_g, 3, None, criteria=critera, attempts=10, flags=flags)
    rr, best_r, center_r = cv2.kmeans(data_r, 3, None, criteria=critera, attempts=10, flags=flags)


    center_b = np.uint8(center_b)
    center_g = np.uint8(center_g)
    center_r = np.uint8(center_r)


    # Make sure you select the area yourself, using the average pixel to determine the label area
    b0 = (best_b.ravel() == 0)
    b1 = (best_b.ravel() == 1)
    b2 = (best_b.ravel() == 2)
    b_0 = np.mean(data_b[best_b.ravel() == 0])
    b_1 = np.mean(data_b[best_b.ravel() == 1])
    b_2 = np.mean(data_b[best_b.ravel() == 2])
    if b_0 >= b_1 and b_0 >= b_2:
        max_b = b0
        if b_1 >= b_2:
            mid_b = b1
        else:
            mid_b = b2
    elif b_1 >= b_2 and b_1 >= b_0:
        max_b = b1
        if b_2 >= b_0:
            mid_b = b2
        else:
            mid_b = b0
    elif b_2 >= b_0 and b_2 >= b_1:
        max_b = b2
        if b_0 >= b_1:
            mid_b = b0
        else:
            mid_b = b1

    g0 = best_g.ravel() == 0
    g1 = best_g.ravel() == 1
    g2 = best_g.ravel() == 2
    g_0 = np.mean(data_g[best_g.ravel() == 0])
    g_1 = np.mean(data_g[best_g.ravel() == 1])
    g_2 = np.mean(data_g[best_g.ravel() == 2])
    if g_0 >= g_1 and g_0 >= g_2:
        max_g = g0
        if g_1 >= g_2:
            mid_g = g1
        else:
            mid_g = g2
    elif g_1 >= g_2 and g_1 >= g_0:
        max_g = g1
        if g_2 >= g_0:
            mid_g = g2
        else:
            mid_g = g0
    elif g_2 >= g_0 and g_2 >= g_1:
        max_g = g2
        if g_0 >= g_1:
            mid_g = g0
        else:
            mid_g = g1

    r0 = best_r.ravel() == 0
    r1 = best_r.ravel() == 1
    r2 = best_r.ravel() == 2
    r_0 = np.mean(data_r[best_r.ravel() == 0])
    r_1 = np.mean(data_r[best_r.ravel() == 1])
    r_2 = np.mean(data_r[best_r.ravel() == 2])
    if r_0 >= r_1 and r_0 >= r_2:
        max_r = r0
        if r_1 >= r_2:
            mid_r = r1
        else:
            mid_r = r2
    elif r_1 >= r_2 and r_1 >= r_0:
        max_r = r1
        if r_2 >= r_0:
            mid_r = r2
        else:
            mid_r = r0
    elif r_2 >= r_0 and r_2 >= r_1:
        max_r = r2
        if r_0 >= r_1:
            mid_r = r0
        else:
            mid_r = r1

    # The data of different categories are re-endowed with another color to achieve segmentation of the pictur
    comp_b = (gray_b_BB.ravel() >= 215) & (max_b)
    mean_b = np.mean(data_b[max_b])
    print(mean_b)
    data_b[comp_b] = (mean_b, mean_b, mean_b)

    comp_g = (gray_g_GG.ravel() >= 205) & (max_g)
    mean_g = np.mean(data_g[max_g])
    print(mean_g)
    data_g[comp_b] = (mean_g, mean_g, mean_g)

    comp_r = (gray_r_RR.ravel() >= 215) & (max_r)
    mean_r = np.mean(data_r[max_r])
    print(mean_r)
    data_r[comp_b] = (mean_r, mean_r, mean_r)



    # Convert the result to the desired format for the picture
    data_b = np.uint8(data_b)
    oi_b = data_b.reshape((img.shape))
    data_g = np.uint8(data_g)
    oi_g = data_g.reshape((img.shape))
    data_r = np.uint8(data_r)
    oi_r = data_r.reshape((img.shape))

    # Convert grayscale images to single-channel images and merge them into color images
    new_b = oi_b[:, :, 0]
    new_g = oi_g[:, :, 1]
    new_r = oi_r[:, :, 2]
    new_img = cv2.merge([new_b, new_g, new_r])

    # Build a complete path to the output image
    new_img =cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)
    new_img = Image.fromarray(new_img)
    pathoutput = os.path.join(path_output, imagePath[27:])
    new_img.save(pathoutput)



