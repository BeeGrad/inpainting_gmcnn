import os
import cv2
import numpy as np
from util.utils import generate_rect_mask, generate_stroke_mask
from model.net import InpaintingModel_GMCNN
from options.options import TestOptions

path_in = 'imgs/celebahq_256x256/'
path_out = 'results/celebahq_256x256/'

images = os.listdir(path_in)

config = TestOptions().parse()

model = InpaintingModel_GMCNN(in_channels=4, opt=config)
model.load_networks(config.load_model_dir)

for img_file in images:
    image = cv2.imread(path_in + img_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, c = image.shape
    mask, _ = generate_rect_mask(im_size=[h, w, c], mask_size=[128, 128])
    # mask = generate_stroke_mask(im_size=[h, w, c])

    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    
    input_img = np.transpose(image[0][::-1,:,:], [1, 2, 0])
    
    image_vis = image * (1 - mask) + 255 * mask
    image_vis = np.transpose(image_vis[0][::-1,:,:], [1, 2, 0])
    
    result = model.evaluate(image, mask)
    result = np.transpose(result[0][::-1,:,:], [1, 2, 0])

    cv2.imwrite(str(path_out + 'original_' + img_file), input_img)
    cv2.imwrite(str(path_out + 'masked_' + img_file), image_vis.astype(np.uint8))
    cv2.imwrite(str(path_out + 'output_' + img_file), result)

    print(img_file, 'saved.')

print('done.')
