import pickle
import random
from collections import Counter

import imgaug as ia  # https://github.com/aleju/imgaug
import numpy as np

from settings import *

DESIRED_SAMPLES_IN_CATEGORY = 500  # TODO Maybe augment to highest number of samples
X = np.array(pickle.load(open(X_PICKLED, "rb")))
y = np.array(pickle.load(open(Y_PICKLED, "rb")))
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


img_transformations_list = {
    random_rotation,
    random_noise,
    random_blur
}

# PLOTSDIR = '../plots/'
# cv2.imwrite(f'{PLOTSDIR}test.jpg', X[0])
# cv2.imwrite(f'{PLOTSDIR}rotation.jpg', random_rotation(X[0]))
# cv2.imwrite(f'{PLOTSDIR}noise.jpg', random_noise(X[0]))
# cv2.imwrite(f'{PLOTSDIR}blur.jpg', random_blur(X[0]))
# cv2.imwrite(f'{PLOTSDIR}snow.jpg', random_snowflakes(X[0]))
# cv2.imwrite(f'{PLOTSDIR}fog.jpg', random_fog(X[0]))
# cv2.imwrite(f'{PLOTSDIR}salt.jpg', random_salt(X[0]))

categories_counter = dict(Counter(y))
print(categories_counter)
aug_y = []
aug_X = []
# cv2.imwrite('TestX.jpg', X[0])
# print(X[0])
for Xi, yi in zip(X, y):
    single_image_transformations = round(DESIRED_SAMPLES_IN_CATEGORY / categories_counter[yi])
    for i in range(single_image_transformations):
        num_transformations_to_apply = random.randint(1, len(img_transformations_list))
        # print(num_transformations_to_apply)
        chosen_transformations = random.sample(img_transformations_list, num_transformations_to_apply)
        # print(chosen_transformations)
        # num_transformations = 0
        transformed_image = None
        for transformation in chosen_transformations:
            transformed_image = transformation(Xi)
        # while num_transformations <= num_transformations_to_apply:
        #     # key = random.choice(list(available_transformations))
        #     # print(key)
        #     # transformed_image = available_transformations[key](Xi)
        #     img_transformations_list[random.choice(list(img_transformations_list))]()
        #     num_transformations += 1
        aug_y.append(yi)
        aug_X.append(transformed_image)
    # break
# print()
print("len aug y: " + str(len(aug_y)))
print("len aug X: " + str(len(aug_X)))
print("len y before merge: " + str(len(y)))
print("len X before merge: " + str(len(X)))
y = np.append(y, aug_y)
X = np.append(X, aug_X, 0)

print("len(y)")
print(len(y))
print("len(X)")
print(len(X))

pickle_out = open(X_PICKLED, "wb")
pickle.dump(X, pickle_out)
pickle_out.close()
pickle_out = open(Y_PICKLED, "wb")
pickle.dump(y, pickle_out)
pickle_out.close()
