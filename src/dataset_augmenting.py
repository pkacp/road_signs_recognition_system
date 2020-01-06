import pickle
import random
import os
import cv2
from collections import Counter

import imgaug as ia  # https://github.com/aleju/imgaug
import numpy as np

from plots_lib import *
from settings import *


def random_rotation(img):
    rotate = ia.augmenters.Affine(rotate=(-25, 25))
    return rotate.augment_image(img)


def random_noise(img):
    noise = ia.augmenters.AdditiveGaussianNoise(scale=(5, 20))
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


def flip_vertically(img):
    return np.fliplr(img)


img_transformations_list = [
    random_rotation, random_noise, random_blur, random_motion_blur, random_snowflakes, random_fog, random_salt
]


# possible to make also horizontal augmenting flips for some sign types
def augment_with_vertical_flip(possible_categories_to_flip):
    for i in possible_categories_to_flip:
        print(CATEGORIES[i])
    X = np.array(pickle.load(open(X_TRAIN_PICKLED, "rb")))
    y = np.array(pickle.load(open(Y_TRAIN_PICKLED, "rb")))
    aug_y = []
    aug_X = []
    # TODO finish this flip


def augment_each_category_to_size(X, X_save_dir, y, y_save_dir, desired_category_size, list_of_transformations,
                                  title_str):
    if title_str == 'train':
        # Save sample transformations
        cv2.imwrite(f'{PLOTSDIR}test.jpg', X[0])
        cv2.imwrite(f'{PLOTSDIR}rotation.jpg', random_rotation(X[0]))
        cv2.imwrite(f'{PLOTSDIR}noise.jpg', random_noise(X[0]))
        cv2.imwrite(f'{PLOTSDIR}blur.jpg', random_blur(X[0]))
        cv2.imwrite(f'{PLOTSDIR}motion_blur.jpg', random_motion_blur(X[0]))
        cv2.imwrite(f'{PLOTSDIR}snow.jpg', random_snowflakes(X[0]))
        cv2.imwrite(f'{PLOTSDIR}fog.jpg', random_fog(X[0]))
        cv2.imwrite(f'{PLOTSDIR}salt.jpg', random_salt(X[0]))

    categories_counter = dict(Counter(y))
    print(categories_counter)
    aug_y = []
    aug_X = []
    aug_categories_counter = dict(Counter(y))
    for Xi, yi in zip(X, y):
        if aug_categories_counter[yi] > desired_category_size:
            continue
        single_image_transformations = round(desired_category_size / categories_counter[yi])
        for i in range(single_image_transformations):
            num_transformations_to_apply = random.randint(1, 3)
            chosen_transformations = random.sample(list_of_transformations, num_transformations_to_apply)
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
    # Draw barchart with category sizes after augmenting
    categories_counter = dict(Counter(y))
    print(categories_counter)
    bar_chart(categories_counter.values(), CATEGORIES, f"{title_str}_categories_to_quantity_chart_after_augmenting")
    # Draw a chart with sample images
    sample_images = random.sample(list(X), 256)
    image_mosaic(sample_images, f"{title_str}_sample_images_after_augmenting", 'gray')


    pickle_out = open(X_save_dir, "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()
    pickle_out = open(y_save_dir, "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()


X_train = np.array(pickle.load(open(X_TRAIN_PICKLED, "rb")))
y_train = np.array(pickle.load(open(Y_TRAIN_PICKLED, "rb")))

X_validate = np.array(pickle.load(open(X_VAL_PICKLED, "rb")))
y_validate = np.array(pickle.load(open(Y_VAL_PICKLED, "rb")))

# augment_with_vertical_flip(CAN_BE_AUGMENTED_WITH_VERT_FLIP_INDEXES)
# augment_each_category_to_size(X_train, X_TRAIN_PICKLED, y_train, Y_TRAIN_PICKLED, DESIRED_TRAINING_CATEGORY_SIZE,
#                               img_transformations_list, 'train')
# augment_each_category_to_size(X_validate, X_VAL_PICKLED, y_validate, Y_VAL_PICKLED, DESIRED_VALIDATION_CATEGORY_SIZE,
#                               img_transformations_list, 'validation')

X_train = np.array(pickle.load(open(X_TRAIN_PICKLED, "rb")))
y_train = np.array(pickle.load(open(Y_TRAIN_PICKLED, "rb")))

X_validate = np.array(pickle.load(open(X_VAL_PICKLED, "rb")))
y_validate = np.array(pickle.load(open(Y_VAL_PICKLED, "rb")))

# Draw chart with number of images in categories train and val
categories_counter = dict(Counter(y_train))
print(categories_counter)
val_categories_counter = dict(Counter(y_validate))
print(val_categories_counter)
double_bar_chart(categories_counter.values(), 'Zbiór trenujący', val_categories_counter.values(),
                 'Zbiór walidacyjny',
                 CATEGORIES, 'double_set_after_augmenting')

# TODO make script from that for Augmenting left to right signs
# list_of_files = list()
# for (dirpath, dirnames, filenames) in os.walk("/home/piotr/Obrazy2/test_images/Nakaz jazdy po prawej stronie znaku"):
#     list_of_files += [os.path.join(dirpath, file) for file in filenames]
# all_images = []
# i = 0
# for file in list_of_files:
#     cv2.imwrite(f'/home/piotr/Obrazy2/test_images/tmp1/mirror_nakaz_po_prawej_{i}.png', np.fliplr(cv2.imread(file)))
#     i += 1
