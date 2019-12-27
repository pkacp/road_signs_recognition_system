import pickle
import random
from collections import Counter

import imgaug as ia  # https://github.com/aleju/imgaug
import numpy as np

from plots_lib import bar_chart, image_mosaic
from settings import *

DESIRED_SAMPLES_IN_CATEGORY = 500  # TODO Maybe augment to highest number of samples
X = np.array(pickle.load(open(X_PICKLED, "rb")))
y = np.array(pickle.load(open(Y_PICKLED, "rb")))
CATEGORIES_NUMBER = len(np.unique(y))


def random_rotation(img):
    rotate = ia.augmenters.Affine(rotate=(-25, 25))
    return rotate.augment_image(img)


def random_noise(img):
    noise = ia.augmenters.AdditiveGaussianNoise(scale=(10, 50))
    return noise.augment_image(img)


def random_blur(img):
    blur = ia.augmenters.GaussianBlur(sigma=(0.2, 1.8))
    return blur.augment_image(img)


def random_motion_blur(img):
    motion_blur = ia.augmenters.MotionBlur(k=(3, 10), angle=(-90, 90))
    return motion_blur.augment_image(img)


def random_snowflakes(img):
    snowflakes = ia.augmenters.Snowflakes()
    return snowflakes.augment_image(img)


def random_fog(img):
    fog = ia.augmenters.Fog()
    return fog.augment_image(img)


def random_salt(img):
    coarse_salt_and_pepper = ia.augmenters.CoarseSaltAndPepper(p=(0.05, 0.3), size_percent=(0.01, 0.5))
    return coarse_salt_and_pepper.augment_image(img)


img_transformations_list = [
    random_rotation, random_noise, random_blur, random_motion_blur, random_snowflakes, random_fog, random_salt
]

# PLOTSDIR = '../plots/'
# cv2.imwrite(f'{PLOTSDIR}test.jpg', X[0])
# cv2.imwrite(f'{PLOTSDIR}rotation.jpg', random_rotation(X[0]))
# cv2.imwrite(f'{PLOTSDIR}noise.jpg', random_noise(X[0]))
# cv2.imwrite(f'{PLOTSDIR}blur.jpg', random_blur(X[0]))
# cv2.imwrite(f'{PLOTSDIR}motion_blur.jpg', random_motion_blur(X[0]))
# cv2.imwrite(f'{PLOTSDIR}snow.jpg', random_snowflakes(X[0]))
# cv2.imwrite(f'{PLOTSDIR}fog.jpg', random_fog(X[0]))
# cv2.imwrite(f'{PLOTSDIR}salt.jpg', random_salt(X[0]))

categories_counter = dict(Counter(y))
print("Number of sample images in categories before augmenting:")
print(categories_counter)
aug_y = []
aug_X = []
aug_categories_counter = dict(Counter(y))
for Xi, yi in zip(X, y):
    if aug_categories_counter[yi] > DESIRED_SAMPLES_IN_CATEGORY:
        continue
    single_image_transformations = round(DESIRED_SAMPLES_IN_CATEGORY / categories_counter[yi])
    for i in range(single_image_transformations):
        num_transformations_to_apply = random.randint(1, int(len(img_transformations_list) / 2))
        chosen_transformations = random.sample(img_transformations_list, num_transformations_to_apply)
        transformed_image = None
        for transformation in chosen_transformations:
            transformed_image = transformation(Xi)
        aug_y.append(yi)
        aug_X.append(transformed_image)
        aug_categories_counter[yi] += 1

print("len aug y: " + str(len(aug_y)))
print("len aug X: " + str(len(aug_X)))
print("len y before merge: " + str(len(y)))
print("len X before merge: " + str(len(X)))
y = np.append(y, aug_y)
X = np.append(X, aug_X, 0)

print("len(y) after merge:")
print(len(y))
print("len(X) after merge:")
print(len(X))

categories_counter = dict(Counter(y))
print(categories_counter)
bar_chart(categories_counter.values(), CATEGORIES,
          "Wykres ilości znaków w poszczególnych kategoriach\npo operacji sztucznego pogłębienia danych trenujących")
# Draw a chart with sample images
sample_images = random.sample(list(X), 256)
image_mosaic(sample_images, "Przykładowe obazy po operacjach pogłębiania zbioru danych trenujących")

pickle_out = open(X_PICKLED, "wb")
pickle.dump(X, pickle_out)
pickle_out.close()
pickle_out = open(Y_PICKLED, "wb")
pickle.dump(y, pickle_out)
pickle_out.close()
