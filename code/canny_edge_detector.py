# Solves problem 4(c)
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import feature
from sklearn.metrics import adjusted_rand_score

# Canny edge detection on the training images because only their labels are available

X_train_raw = []
y_train_raw = []
X_test_raw = []
# load data from the .tiff files
for i in range(1,31):
	im_train = Image.open('data/segmentation_data/trainvolume'+str(i)+'.tiff')
	X_train_raw.append(np.array(im_train, dtype = theano.config.floatX))
	im_label = Image.open('data/segmentation_data/trainlabels'+str(i)+'.tiff')
	y_train_raw.append(np.array(im_label, dtype = theano.config.floatX))
	im_test = Image.open('data/segmentation_data/testvolume'+str(i)+'.tiff')
	X_test_raw.append(np.array(im_test, dtype = theano.config.floatX))
X_train_raw = X_train_raw.reshape((X_train_raw.shape[0],1,512,512)) # (30,1,512,512) 30 training images of dimension 512 * 512
y_train_raw = y_train_raw.reshape((y_train_raw.shape[0],1,512,512)) # (30,1,512,512) 30 training label images of dimension 512 * 512
y_train = y_train_raw.reshape((7864320,)) # flatten the labels for comparison

test_pred = []
for i in range(30):
	np.append(test_pred, feature.canny(X_train_raw[i,1,:,:], sigma=2).reshape((262144,))))

# pixel error = total proportion of pixels that are wrongly classified
pixel_err = (np.where((test_pred - y_test) != 0).shape[0]) / (30 * 512 ** 2)
# Rand error = 1 - rand index/score
# rank index = (a + b) / (n choose 2)
# where a = the number of pixels that are in the same segment in test_pred and in the same segment in y_test
# and b = the number of pixels that are in different segments in test_pred and in different segments in y_test
rand_err = 1 - adjusted_rand_score(y_test, test_pred)