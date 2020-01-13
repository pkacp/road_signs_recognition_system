import os
import pickle
import random
from collections import Counter

import cv2
import numpy as np

from plots_lib import bar_chart, image_mosaic, double_bar_chart
from settings import *


def create_dataset_gray(main_dir, type):
    training_dataset = []
    X = []
    y = []
    for category_name in CATEGORIES:
        category_path = os.path.join(main_dir, category_name)
        category_number = CATEGORIES.index(category_name)
        for img in os.listdir(category_path):
            try:
                training_dataset.append([cv2.resize(cv2.imread(os.path.join(category_path, img), cv2.IMREAD_GRAYSCALE),
                                                    (IMG_WIDTH, IMG_HEIGHT)), category_number])
            except Exception as e:
                print(e)
    # reshape X
    for array, label in training_dataset:
        X.append(array)
        y.append(label)
    X = np.array(X).reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)
    print(y)
    # Draw chart for numbers of categories
    categories_counter = dict(Counter(y))
    print(type)
    print(categories_counter)
    bar_chart(categories_counter.values(), CATEGORIES, f"{type}_categories_to_quantity_chart")
    # Draw a chart with sample images
    sample_images = random.sample(list(X), 256)
    image_mosaic(sample_images, f"{type}_sample_images_after_read_in_grayscale_and_resize")
    return X, y


def create_dataset_rgb(main_dir, type):
    training_dataset = []
    X = []
    y = []
    for category_name in CATEGORIES:
        category_path = os.path.join(main_dir, category_name)
        category_number = CATEGORIES.index(category_name)
        for img in os.listdir(category_path):
            try:
                training_dataset.append([cv2.resize(cv2.imread(os.path.join(category_path, img)),
                                                    (IMG_WIDTH, IMG_HEIGHT)), category_number])
            except Exception as e:
                print(e)
    # return training_dataset
    for array, label in training_dataset:
        X.append(array)
        y.append(label)
    X = np.array(X).reshape(-1, IMG_HEIGHT, IMG_WIDTH, 3)
    print(y)
    # Draw chart for numbers of categories
    categories_counter = dict(Counter(y))
    print(type)
    print(categories_counter)
    bar_chart(categories_counter.values(), CATEGORIES, f"{type}_categories_to_quantity_chart")
    # Draw a chart with sample images
    sample_images = random.sample(list(X), 256)
    image_mosaic(sample_images, f"{type}_sample_images_after_read_in_rgb_and_resize")
    return X, y


def save_to_picle(X, X_dir, y, y_dir):
    pickle_out = open(X_dir, "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()
    pickle_out = open(y_dir, "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()


# Draw chart of sample images
list_of_files = list()
for (dirpath, dirnames, filenames) in os.walk(IMAGES_TRAIN_DIR):
    list_of_files += [os.path.join(dirpath, file) for file in filenames]
all_images = []
for file in list_of_files:
    all_images.append(cv2.imread(file))
sample_images = random.sample(list(all_images), 256)
image_mosaic(sample_images, "sample_images")

if COLOR_MODE == 'gray':
    X, y = create_dataset_gray(IMAGES_TRAIN_DIR, 'train')
    X_val, y_val = create_dataset_gray(IMAGES_VAL_DIR, 'val')
    X_test, y_test = create_dataset_gray(IMAGES_TEST_DIR, 'test')
elif COLOR_MODE == 'rgb':
    X, y = create_dataset_rgb(IMAGES_TRAIN_DIR, 'train')
    X_val, y_val = create_dataset_rgb(IMAGES_VAL_DIR, 'val')
    X_test, y_test = create_dataset_rgb(IMAGES_TEST_DIR, 'test')

# Draw chart with number of images in categories train and val
categories_counter = dict(Counter(y))
val_categories_counter = dict(Counter(y_val))
double_bar_chart(categories_counter.values(), 'Zbiór trenujący', val_categories_counter.values(),
                 'Zbiór walidacyjny',
                 CATEGORIES, 'double_set_after_read', 'Kategoria znaku', 'Liczba przykładów w kategorii')

save_to_picle(X, X_TRAIN_PICKLED, y, Y_TRAIN_PICKLED)
save_to_picle(X_val, X_VAL_PICKLED, y_val, Y_VAL_PICKLED)
save_to_picle(X_test, X_TEST_PICKLED, y_test, Y_TEST_PICKLED)
