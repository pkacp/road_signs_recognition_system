import os
import pickle
import random
import time
from settings import CATEGORIES

import cv2
import numpy as np

# IMAGESDIR = "/home/piotr/Obrazy" #pc
IMAGESDIR = "/home/piokac/Dokumenty/!inzynierka/Obrazy"  # laptop
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
    # random.shuffle(training_dataset) # moved shuffling to augmenting
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
print(training_dataset[0][0])
cv2.imwrite('Test.jpg', training_dataset[0][0])
X, y = reshape_dataset(training_dataset)
cv2.imwrite('TestX.jpg', X[0])
pickle_out = open(f"{PICKLEDIR}/X_{int(time.time())}.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()
pickle_out = open(f"{PICKLEDIR}/y_{int(time.time())}.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()
