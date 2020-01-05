import cv2
import pickle
import tensorflow as tf
import numpy as np
from settings import *
from plots_lib import image_mosaic


def prepare_image(img_path):
    modified_img = cv2.resize(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), (IMG_WIDTH, IMG_HEIGHT))
    return modified_img.reshape(-1, IMG_WIDTH, IMG_HEIGHT, 1) / 255.0


X_val = np.array(pickle.load(open(X_VAL_PICKLED, "rb")))
y_val = np.array(pickle.load(open(Y_VAL_PICKLED, "rb")))

X_test = np.array(pickle.load(open(X_TEST_PICKLED, "rb")))
y_test = np.array(pickle.load(open(Y_TEST_PICKLED, "rb")))

X_val = X_val / 255.0
X_test = X_test / 255.0

model_name = '1578090785-road_signs_recognition-conv-32(3,3)2x2-64(3,3)2x2-128(3,3)2x2-dense-64-epochs-5.model'
model = tf.keras.models.load_model(f'../saved_models/{model_name}')

# print(X_test[0].shape)
# random_img = np.random.rand(1,32, 32, 1)
# print(random_img.shape)

wong_images = []

# for image, category in zip(X_test, y_test):
predictions = model.predict(X_test)
for prediction, image, fact in zip(predictions, X_test, y_test):
    if np.argmax(prediction) != fact:
        np.set_printoptions(suppress=True)  # suppress scientific print
        print(np.max(prediction))
        print(f"Error in category {fact}")
        wong_images.append(image)

image_mosaic(wong_images, 'wrong_predictions_images', 'gray')
# prediction = model.predict([random_img])
# print(prediction[0]
# print(prediction)

print("-----------------------------------------------------------------------------")
print('\n# Evaluate on test data')
results = model.evaluate(X_test, y_test)
print('test loss, test acc:', results)
