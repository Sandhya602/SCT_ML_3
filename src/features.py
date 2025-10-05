import cv2
import numpy as np
from skimage.feature import hog




def extract_hog_features(image, resize_shape=(128, 128)):
img_resized = cv2.resize(image, resize_shape)
gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
cells_per_block=(2, 2), block_norm='L2-Hys',
transform_sqrt=True, feature_vector=True)
return features




def extract_color_histogram(image, bins=(8, 8, 8)):
hist = cv2.calcHist([image], [0, 1, 2], None, bins,
[0, 256, 0, 256, 0, 256])
cv2.normalize(hist, hist)
return hist.flatten()




def extract_features(image_path):
image = cv2.imread(image_path)
if image is None:
raise ValueError(f"Unable to read image: {image_path}")
hog_feat = extract_hog_features(image)
color_feat = extract_color_histogram(image)
return np.hstack((hog_feat, color_feat))
