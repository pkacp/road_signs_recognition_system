import pickle

import cv2
import imgaug as ia  # https://github.com/aleju/imgaug
import numpy as np

DESIRED_SAMPLES_IN_CATEGORY = 500
X = np.array(pickle.load(open("../pickled_datasets/X.pickle", "rb")))
y = np.array(pickle.load(open("../pickled_datasets/y.pickle", "rb")))
CATEGORIES_NUMBER = len(np.unique(y))


def random_rotation(img):
    rotate = ia.augmenters.Affine(rotate=(-25, 25))
    return rotate.augment_image(img)


def random_noise(img):
    noise = ia.augmenters.AdditiveGaussianNoise(scale=(10, 60))
    return noise.augment_image(img)


def random_blur(img):
    blur = ia.augmenters.GaussianBlur(sigma=(0.2, 1.8))
    return blur.augment_image(img)


def random_snowflakes(img):
    snowflakes = ia.augmenters.Snowflakes()
    return snowflakes.augment_image(img)


def random_fog(img):
    fog = ia.augmenters.Fog()
    return fog.augment_image(img)


def random_salt(img):
    coarse_salt_and_pepper = ia.augmenters.CoarseSaltAndPepper(p=(0.05, 0.3), size_percent=(0.01, 0.5))
    return coarse_salt_and_pepper.augment_image(img)


available_transformations = {
    'rotate': random_rotation,
    'noise': random_noise
}

PLOTSDIR = '../plots/'
cv2.imwrite(f'{PLOTSDIR}test.jpg', X[0])
cv2.imwrite(f'{PLOTSDIR}rotation.jpg', random_rotation(X[0]))
cv2.imwrite(f'{PLOTSDIR}noise.jpg', random_noise(X[0]))
cv2.imwrite(f'{PLOTSDIR}blur.jpg', random_blur(X[0]))
cv2.imwrite(f'{PLOTSDIR}snow.jpg', random_snowflakes(X[0]))
cv2.imwrite(f'{PLOTSDIR}fog.jpg', random_fog(X[0]))
cv2.imwrite(f'{PLOTSDIR}salt.jpg', random_salt(X[0]))

# categories_counter = dict(Counter(y))
# print(categories_counter)
# aug_y = []
# aug_X = []
# # cv2.imwrite('TestX.jpg', X[0])
# print(X[0])
# for Xi, yi in zip(X, y):
#     single_image_transformations = round(DESIRED_SAMPLES_IN_CATEGORY / categories_counter[yi])
#     for i in range(single_image_transformations):
#         num_transformations_to_apply = random.randint(0, len(available_transformations))
#         print(num_transformations_to_apply)
#         num_transformations = 0
#         transformed_image = None
#         while num_transformations <= num_transformations_to_apply:
#             # key = random.choice(list(available_transformations))
#             # print(key)
#             # transformed_image = available_transformations[key](Xi)
#             available_transformations[random.choice(list(available_transformations))]()
#             num_transformations += 1
#         aug_y.append(yi)
#         aug_X.append(transformed_image)
#     # break
# print()
# print("len aug y: " + str(len(aug_y)))
# # print(aug_X)
# print("len y before merge: " + str(len(y)))
# y = np.append(y, aug_y)
# X = np.append(X, aug_X)
#
# print(len(y))
