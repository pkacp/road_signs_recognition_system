import os
import pickle
import random
from collections import Counter

import cv2
import numpy as np

from plots_lib import bar_chart, image_mosaic
from settings import *


def create_dataset():
    training_dataset = []
    for category_name in CATEGORIES:
        category_path = os.path.join(IMAGES_BASE_DIR, category_name)
        category_number = CATEGORIES.index(category_name)
        for img in os.listdir(category_path):
            try:
                training_dataset.append([cv2.resize(cv2.imread(os.path.join(category_path, img), cv2.IMREAD_GRAYSCALE),
                                                    (IMG_WIDTH, IMG_HEIGHT)), category_number])
            except Exception as e:
                print(e)
    return training_dataset


def reshape_dataset(dataset):
    X = []
    y = []
    for array, label in dataset:
        X.append(array)
        y.append(label)
    X = np.array(X).reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)
    print(y)
    return X, y


# Draw chart of sample images
list_of_files = list()
for (dirpath, dirnames, filenames) in os.walk(IMAGES_BASE_DIR):
    list_of_files += [os.path.join(dirpath, file) for file in filenames]
all_images = []
for file in list_of_files:
    all_images.append(cv2.resize(cv2.imread(file), (50, 50)))
sample_images = random.sample(list(all_images), 256)
image_mosaic(sample_images, "sample_images", 'rgb')

training_dataset = create_dataset()
X, y = reshape_dataset(training_dataset)

# Draw chart for numbers of categories
categories_counter = dict(Counter(y))
print(categories_counter)
bar_chart(categories_counter.values(), CATEGORIES, "categories_to_quantity_chart")
# Draw a chart with sample images
sample_images = random.sample(list(X), 256)
image_mosaic(sample_images, "sample_images_after_read_in_grayscale_and_resize", 'gray')

pickle_out = open(X_PICKLED, "wb")
pickle.dump(X, pickle_out)
pickle_out.close()
pickle_out = open(Y_PICKLED, "wb")
pickle.dump(y, pickle_out)
pickle_out.close()
