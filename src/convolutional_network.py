import pickle
import random
import time

import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from sklearn.utils import shuffle
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential

# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# tensorboard --logdir="logs"

X = np.array(pickle.load(open("../pickled_datasets/X.pickle", "rb")))
y = np.array(pickle.load(open("../pickled_datasets/y.pickle", "rb")))
categories_number = len(np.unique(y))
print(f"Number of categories: {categories_number}")
print(f"Number of all training images: {len(y)}")

X = X / 255.0
X, y = shuffle(X, y)

NAME = f"road-signs-recognition-conv-net-set-len-{len(y)}-64x64-{int(time.time())}"
tensorboard = TensorBoard(log_dir=f'../logs/{NAME}')

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(categories_number))
model.add(Activation("sigmoid"))

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

model.fit(X, y, batch_size=16, epochs=10, validation_split=0.3, callbacks=[tensorboard])

model.save(f"../saved_models/{NAME}.model")
