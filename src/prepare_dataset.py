import os
import pickle
import random

import cv2
import numpy as np

IMAGESDIR = "/home/piotr/Obrazy"
PICKLEDIR = "../pickled_datasets"
CATEGORIES = ["20", "30", "40", "50", "70", "ustap pierwszenstwa", "koniec pierwszenstwa", "masz_pierwszenstwo",
              "przejscie dla pieszych",
              "stop", "rondo", "zakaz zatrzymywania", "zakaz wjazdu", "zakaz ruchu"]
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
    random.shuffle(training_dataset)
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
X, y = reshape_dataset(training_dataset)
pickle_out = open(f"{PICKLEDIR}/X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()
pickle_out = open(f"{PICKLEDIR}/y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()
