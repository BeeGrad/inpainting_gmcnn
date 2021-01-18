import os
import cv2
import csv
import math
import numpy as np
from skimage.metrics import structural_similarity

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))



path_in = 'validation_randomlines/'
path_or = 'imgs/validation/'

file = open('randomlines_validation_results.csv', 'w', newline='')
writer = csv.writer(file)
header = "Image PSNR SSIM"
writer.writerow(header.split())

i = 0
images = os.listdir(path_in)
for filename in images:
    original_image = cv2.imread(path_or + filename)
    input_image = cv2.imread(path_in + filename)

    psnr = calculate_psnr(original_image, input_image)
    ssim = structural_similarity(original_image, input_image, data_range=1, win_size=11, multichannel=True)

    writer.writerow([filename, psnr, ssim])
    i += 1
    print(i)