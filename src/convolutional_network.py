import pickle
import random
import time

import numpy as np
from settings import *
from tensorflow.keras.callbacks import TensorBoard
from sklearn.utils import shuffle
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2

# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# tensorboard --logdir="logs"

X_test = np.array(pickle.load(open(X_TEST_PICKLED, "rb")))
y_test = np.array(pickle.load(open(Y_TEST_PICKLED, "rb")))

X = np.array(pickle.load(open(X_TRAIN_PICKLED, "rb")))
y = np.array(pickle.load(open(Y_TRAIN_PICKLED, "rb")))
categories_number = len(np.unique(y))
print(f"Number of categories: {categories_number}")
print(f"Number of all training images: {len(y)}")

print(y)
X, y = shuffle(X, y)
print(y)

X = X / 255.0

NAME = f"road_signs_recognition-conv-additional_dense_layer-5epochs-{'32(3,3)2x2-64(3,3)2x2-128(3,3)2x2'}-dense-{0}-set_len-{len(y)}-img_size-{'32x32'}-{int(time.time())}"
print(NAME)
tensorboard = TensorBoard(log_dir=f'../logs/{NAME}')

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(categories_number))
model.add(Activation("sigmoid"))

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["sparse_categorical_accuracy"])

model.fit(X, y, batch_size=64, epochs=5, validation_split=0.3, callbacks=[tensorboard])

print('\n# Evaluate on test data')
results = model.evaluate(X_test, y_test, batch_size=8)
print('test loss, test acc:', results)

model.save(f"../saved_models/{NAME}.model")

# dense_layers = [0, 1, 2]
# layer_sizes = [32, 64, 128]
# conv_layers_number = [1, 2, 3]

# dense_layers = [1]
# layer_sizes = [32, 64, 128]
# conv_layers_number = [1, 3]
#
# for dense_layer in dense_layers:
#     for layer_size in layer_sizes:
#         for conv_layer in conv_layers_number:
#             NAME = f"road_signs_recognition-conv-{conv_layer}-nodes-{layer_size}-dense-{dense_layer}-set_len-{len(y)}-{int(time.time())}"
#             print(NAME)
#             tensorboard = TensorBoard(log_dir=f'../logs/{NAME}')
#
#             model = Sequential()
#
#             model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
#             model.add(Activation("relu"))
#             model.add(MaxPooling2D(pool_size=(2, 2)))
#             for layer in range(conv_layer - 1):
#                 model.add(Conv2D(layer_size, (3, 3)))
#                 model.add(Activation("relu"))
#                 model.add(MaxPooling2D(pool_size=(2, 2)))
#
#             model.add(Flatten())
#
#             for dense in range(dense_layer):
#                 model.add(Dense(layer_size))
#                 model.add(Activation("relu"))
#
#             model.add(Dense(categories_number))
#             model.add(Activation("sigmoid"))
#
#             model.compile(loss="sparse_categorical_crossentropy",
#                           optimizer="adam",
#                           metrics=["accuracy"])
#
#             model.fit(X, y, batch_size=16, epochs=10, validation_split=0.3, callbacks=[tensorboard])
#
#             model.save(f"../saved_models/{NAME}.model")
