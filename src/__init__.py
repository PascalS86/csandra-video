import csandra_video.utils as utils
import cv2
import numpy as np

#model = utils.download_tfjs_model(n_modelpath='bodypix/resnet50/float/model-stride16')
graph = utils.load_graph_model(n_modelpath='models/bodypix/resnet50/float/model_stride16/model.json')
img = cv2.imread('test.jpg')
#result = utils.eval_image(n_graph=graph, n_image=img)
mask = utils.get_segementation_mask(n_graph=graph, n_image=img)

# read in a "virtual background" (should be in 16:9 ratio)
replacement_bg_raw = cv2.imread('backgroundtest.jpg')

# resize to match the frame (width & height from before)
height, width = 720, 1280
replacement_bg = cv2.resize(replacement_bg_raw, (width, height))

# combine the background and foreground, using the mask and its inverse
inv_mask = np.bitwise_not(mask)
print(inv_mask.shape)
cv2.imwrite('color_img.jpg', inv_mask)
f = np.bitwise_and(img, mask)
inv_f = np.bitwise_and(replacement_bg, inv_mask)
f = f + inv_f

cv2.imwrite('color_img.jpg', f)
