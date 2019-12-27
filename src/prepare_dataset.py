import os
import pickle
from collections import Counter

import cv2
import numpy as np

from plots_lib import bar_chart
from settings import CATEGORIES

IMAGESDIR = "/home/piotr/Obrazy"  # pc
# IMAGESDIR = "/home/piokac/Dokumenty/!inzynierka/Obrazy"  # laptop
PICKLEDIR = "../pickled_datasets"

IMG_WIDTH = 50
IMG_HEIGHT = 50


def create_dataset():
    training_dataset = []
    for category_name in CATEGORIES:
        category_path = os.path.join(IMAGESDIR, category_name)
        category_number = CATEGORIES.index(category_name)
        for img in os.listdir(category_path):
            try:
                training_dataset.append([cv2.resize(cv2.imread(os.path.join(category_path, img), cv2.IMREAD_GRAYSCALE),
                                                    (IMG_WIDTH, IMG_HEIGHT)), category_number])
            except Exception as e:
                print(e)
    # random.shuffle(training_dataset) # moved shuffling after augmenting
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


training_dataset = create_dataset()
# cv2.imwrite('Test.jpg', training_dataset[0][0])
X, y = reshape_dataset(training_dataset)
# Draw chart for numbers of categories
categories_counter = dict(Counter(y))
print(categories_counter)
bar_chart(categories_counter.values(), CATEGORIES, "Wykres ilości znaków w poszczególnych kategoriach")

# cv2.imwrite('TestX.jpg', X[0])
pickle_out = open(f"{PICKLEDIR}/X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()
pickle_out = open(f"{PICKLEDIR}/y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

# Working synchronous shuffle
# a = [["a", 2, 3, 4, 5], ["b", 2, 3, 4, 5], ["c", 2, 3, 4, 5]]
# b = ["a", "b", "c"]
# temp = list(zip(a, b))
# random.shuffle(temp)
# a, b = zip(*temp)
# print(a, b)
