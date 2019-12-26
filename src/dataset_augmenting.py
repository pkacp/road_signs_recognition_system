import pickle
import random
import cv2

# import imgaug as ia  # https://github.com/aleju/imgaug
import numpy as np
# from scipy import ndarray
# import skimage as sk
# from skimage import filters
# from skimage import transform
from collections import Counter

DESIRED_SAMPLES_IN_CATEGORY = 500
X = np.array(pickle.load(open("../pickled_datasets/X.pickle", "rb")))
y = np.array(pickle.load(open("../pickled_datasets/y.pickle", "rb")))
CATEGORIES_NUMBER = len(np.unique(y))


# def random_rotation(image_array: ndarray):
#     random_degree = random.uniform(-25, 25)
#     return sk.transform.rotate(image_array, random_degree)
#
#
# def random_blur(image_array: ndarray):
#     random_value = random.uniform(0, 3)
#     return sk.filters.gaussian(image_array, random_value)
#
#
# def random_noise(image_array: ndarray):
#     return sk.util.random_noise(image_array)
#


def random_rotation():
    print("random_rotation")


def random_noise():
    print("random_noise")


def random_blur():
    print("random_blur")


available_transformations = {
    'rotate': random_rotation,
    'noise': random_noise,
    'blur': random_blur
}

categories_counter = dict(Counter(y))
print(categories_counter)
aug_y = []
aug_X = []
# cv2.imwrite('TestX.jpg', X[0])
for Xi, yi in zip(X, y):
    single_image_transformations = round(DESIRED_SAMPLES_IN_CATEGORY / categories_counter[yi])
    for i in range(single_image_transformations):
        num_transformations_to_apply = random.randint(0, len(available_transformations))
        print(num_transformations_to_apply)
        num_transformations = 0
        transformed_image = None
        while num_transformations <= num_transformations_to_apply:
            # key = random.choice(list(available_transformations))
            # print(key)
            # transformed_image = available_transformations[key](Xi)
            available_transformations[random.choice(list(available_transformations))]()
            num_transformations += 1
        aug_y.append(yi)
        aug_X.append(transformed_image)
    # break

print("len aug y: " + str(len(aug_y)))
# print(aug_X)
print("len y before merge: " + str(len(y)))
y = np.append(y, aug_y)
X = np.append(X, aug_X)

print(len(y))
