import pickle
import random
import time

import numpy as np
from settings import *
from plots_lib import *
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from sklearn.utils import shuffle
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential

# from tensorflow.keras import optimizers
# from tensorflow.keras import losses
# from tensorflow.keras import metrics
# from tensorflow.keras import regularizers

# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# tensorboard --logdir="logs"

X_test = np.array(pickle.load(open(X_TEST_PICKLED, "rb")))
y_test = np.array(pickle.load(open(Y_TEST_PICKLED, "rb")))

X_val = np.array(pickle.load(open(X_VAL_PICKLED, "rb")))
y_val = np.array(pickle.load(open(Y_VAL_PICKLED, "rb")))

X_train = np.array(pickle.load(open(X_TRAIN_PICKLED, "rb")))
y_train = np.array(pickle.load(open(Y_TRAIN_PICKLED, "rb")))

categories_number = len(np.unique(y_train))
print(f"Number of categories: {categories_number}")
print(f"Number of all training images: {len(y_train)}")

print(y_train)
X_train, y_train = shuffle(X_train, y_train)
X_val, y_val = shuffle(X_val, y_val)
print(y_train)

X_train = X_train / 255.0
X_val = X_val / 255.0
X_test = X_test / 255.0

history_arr = []

NAME = f"{int(time.time())}-rgb-road_signs_recognition-conv-{'64(3,3)2x2'}-dense-{'64'}-epochs-{10}"
# NAME = f"{int(time.time())}-road_signs_recognition-conv-{0}-dense-{'128x128x64'}-epochs-{10}"
print(NAME)

tensorboard = TensorBoard(log_dir=f'../logs/{NAME}')
early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-2, patience=2, verbose=1)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=X_train.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(64, (3, 3)))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(64, (3, 3)))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

# model.add(Dense(128))
# model.add(Activation("relu"))
#
# model.add(Dense(64))
# model.add(Activation("relu"))

# model.add(Dense(64))
# model.add(Activation("relu"))

model.add(Dense(categories_number))
model.add(Activation("sigmoid"))

loss_function = "sparse_categorical_crossentropy"
optimizer = "adam"
metric = "sparse_categorical_accuracy"

model.compile(loss=loss_function,
              optimizer=optimizer,
              metrics=[metric])

history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    batch_size=32, epochs=10,
                    # callbacks=[early_stopping]
                    )
history_arr.append(history)
model.summary()

print('\n# Evaluate on test data')
results = model.evaluate(X_test, y_test)
print('test loss, test acc:', results)
plot_accuracy_history(history, metric, f'acc_{NAME}')
plot_loss_history(history, loss_function, f'loss_{NAME}')
# model.save(f"../saved_models/{NAME}.model")


print("#####################################################")

NAME = f"{int(time.time())}-road_signs_recognition-dense-{'256x256'}-epochs-{10}"
print(NAME)

model = Sequential()
# model.add(Conv2D(64, (3, 3), input_shape=X_train.shape[1:]))

model.add(Flatten())

# model.add(Dense(512))
# model.add(Activation("relu"))

model.add(Dense(256))
model.add(Activation("relu"))

model.add(Dense(categories_number))
model.add(Activation("sigmoid"))

loss_function = "sparse_categorical_crossentropy"
optimizer = "adam"
metric = "sparse_categorical_accuracy"

model.compile(loss=loss_function,
              optimizer=optimizer,
              metrics=[metric])

history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    batch_size=32, epochs=10,
                    # callbacks=[early_stopping]
                    )

history_arr.append(history)
model.summary()

print('\n# Evaluate on test data')
results = model.evaluate(X_test, y_test)
print('test loss, test acc:', results)
plot_accuracy_history(history, metric, f'acc_{NAME}')
plot_loss_history(history, loss_function, f'loss_{NAME}')

plot_multi_val_accuracy(history_arr, ['Sieć z warstwą konwolucyjną', 'Sieć bez warstwy konwolucyjnej'], metric,
                        'conv_vs_no_conv_val_accuracy')
plot_multi_val_loss(history_arr, ['Sieć z warstwą konwolucyjną', 'Sieć bez warstwy konwolucyjnej'],
                    'conv_vs_no_conv_val_loss')
