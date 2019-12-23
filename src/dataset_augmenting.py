import pickle

import imgaug as ia  # https://github.com/aleju/imgaug
import numpy as np

CATEGORY_DESIRED_NUMBER = 5000
X = np.array(pickle.load(open("../pickled_datasets/X.pickle", "rb")))
y = np.array(pickle.load(open("../pickled_datasets/y.pickle", "rb")))

image1 = X[0]
print(image1)

rotate = ia.augmenters.Affine(rotate=(-25, 25))
image1_aug = rotate.augment_image(image1)

print(image1_aug)
