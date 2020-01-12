import pickle
import random
import time

import numpy as np
from settings import *
from plots_lib import *
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from sklearn.utils import shuffle
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential

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
print('Training dataset shape')
print(X_train.shape)

X_train, y_train = shuffle(X_train, y_train)
X_val, y_val = shuffle(X_val, y_val)

X_train = X_train / 255.0
X_val = X_val / 255.0
X_test = X_test / 255.0

list_of_conv_options = [
    [[16, 3]],
    [[16, 5]],
    [[32, 3]],
    [[32, 5]],
    [[16, 5], [32, 3]],
    [[32, 5], [64, 3]],
    [[16, 3], [32, 3], [64, 3]]
]
names_list = ['Sieć 16(3x3)', 'Sieć 16(5x5)', 'Sieć 32(3x3)', 'Sieć 32(5x5)', 'Sieć 16(5x5)x32(3x3)',
              'Sieć 32(5x5)x64(3x3)', 'Sieć 16(3x3)x32(3x3)x64(3x3)']

epochs = 20
history_arr = []

# tensorboard = TensorBoard(log_dir=f'../logs/{NAME}')
early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-2, patience=2, verbose=1)

model = Sequential()

for option in list_of_conv_options:
    for conv in option:
        NAME = f"{int(time.time())}-rgb-road_signs_recognition-conv-{'x'.join(map(str, conv))}-softmax"
        print(NAME)
        filters = conv[0]
        size = conv[1]
        model.add(Conv2D(32, (size, size), input_shape=X_train.shape[1:], padding='Same'))
        model.add(Activation("relu"))

        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(categories_number))
    model.add(Activation("sigmoid"))

    loss_function = "sparse_categorical_crossentropy"
    optimizer = "adam"
    metric = "sparse_categorical_accuracy"

    model.compile(loss=loss_function,
                  optimizer=optimizer,
                  metrics=[metric])

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        batch_size=64, epochs=epochs,
                        callbacks=[early_stopping]
                        )
    history_arr.append(history)

    model.summary()

    print('\n# Evaluate on test data')
    results = model.evaluate(X_test, y_test)
    print('test loss, test acc:', results)

    model.save(f"../saved_models/{NAME}.model")

    plot_accuracy_history(history, metric, f'acc_{NAME}')
    plot_loss_history(history, loss_function, f'loss_{NAME}')

plot_multi_val_accuracy(history_arr, names_list, metric,
                        'acc_different_conv_sizes')
plot_multi_val_loss(history_arr, names_list,
                    'val_loss_different_conv_sizes')
