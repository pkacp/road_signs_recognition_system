import cv2
import pickle
import tensorflow as tf
import numpy as np
from settings import *


def prepare_image(img_path):
    modified_img = cv2.resize(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), (IMG_WIDTH, IMG_HEIGHT))
    return modified_img.reshape(-1, IMG_WIDTH, IMG_HEIGHT, 1) / 255.0


X_test = np.array(pickle.load(open("../pickled_datasets/X_test.pickle", "rb")))
y_test = np.array(pickle.load(open("../pickled_datasets/y_test.pickle", "rb")))

X_test = X_test / 255.0

model_name = '1578089613-road_signs_recognition-conv-32(3,3)2x2-64(3,3)2x2-128(3,3)2x2-dense-64-epochs-30.model'
model = tf.keras.models.load_model(f'../saved_models/{model_name}')

# test_images_path = '../test_images/'
# test_image_name = 'Zrzut ekranu z 2019-12-28 14-47-17.png'
# prepared_image = prepare_image(f'{test_images_path}{test_image_name}')
#
# print(prepared_image[0].shape)
#
# prediction = model.predict([prepared_image])
#
# print(prediction)
# np.set_printoptions(suppress=True)  # suppress scientific print
# print(prediction)
#
# index_number = np.argmax(prediction)
# max_confidence = np.max(prediction)
# print(f"Given image is in prediction: {CATEGORIES[index_number]} with {max_confidence * 100}% confidence")
print("-----------------------------------------------------------------------------")
print('\n# Evaluate on test data')
results = model.evaluate(X_test, y_test)
print('test loss, test acc:', results)
