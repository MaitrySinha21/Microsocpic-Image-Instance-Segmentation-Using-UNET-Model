from tensorflow.keras.models import load_model
import cv2
from smooth_blending import Final_binary_prediction, Final_prediction

# take image ######################
img = cv2.imread('Flask_nucleus/images/n_1.png', 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# img = cv2.imread('satellite_data/Tile 2/images/image_part_008.jpg')
# ht, wd = img.shape[:2]
# img = cv2.resize(img, (510, 510*ht//wd), interpolation=cv2.INTER_NEAREST)

# take model ######################
model = load_model('Flask_nucleus/model/nucleus_small_256_unet_model.hdf5', compile=False)
# model = load_model('models/satellite_256_unet_model_c6.hdf5', compile=False)

# take smooth blender #############
Final_binary_prediction(img, model, patch_size=256, col=(152, 40, 152))
# Final_prediction(img, model, patch_size=256, n_classes=6)

