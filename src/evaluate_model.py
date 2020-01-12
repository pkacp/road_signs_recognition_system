import cv2
import pickle
import tensorflow as tf
import numpy as np
from settings import *
from plots_lib import *


def prepare_image(img_path):
    modified_img = cv2.resize(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), (IMG_WIDTH, IMG_HEIGHT))
    return modified_img.reshape(-1, IMG_WIDTH, IMG_HEIGHT, 1) / 255.0


X_val = np.array(pickle.load(open(X_VAL_PICKLED, "rb")))
y_val = np.array(pickle.load(open(Y_VAL_PICKLED, "rb")))

X_test = np.array(pickle.load(open(X_TEST_PICKLED, "rb")))
y_test = np.array(pickle.load(open(Y_TEST_PICKLED, "rb")))

X_val = X_val / 255.0
X_test = X_test / 255.0

model_name = '1578714434-rgb-road_signs_recognition-conv-32x32x64x128-sigmoid.model'
# model_name = '1578721238-gray-road_signs_recognition-conv-32x32x64x128-sigmoid.model'
model = tf.keras.models.load_model(f'../saved_models/{model_name}')

error_count = 0
prediction_nr = 0
predictions = model.predict(X_test)

error_number_list = [0] * len(CATEGORIES)
all_correct_prediction_procent = [[] for i in range(len(CATEGORIES))]
for prediction, fact in zip(predictions, y_test):
    id_max_prediction_confidence = np.argmax(prediction)
    max_prediction_confidence = np.max(prediction)
    if id_max_prediction_confidence != fact:
        error_number_list[fact] += 1
    elif id_max_prediction_confidence == fact:
        all_correct_prediction_procent[fact].append(max_prediction_confidence)

mean_all_correct_prediction_procent = [0] * len(CATEGORIES)
for idx, category_procent_predictions in enumerate(all_correct_prediction_procent):
    mean_all_correct_prediction_procent[idx] = sum(category_procent_predictions) / len(category_procent_predictions)
bar_chart([i * 100 for i in mean_all_correct_prediction_procent], CATEGORIES, 'mean_percent_of_correct_predictions_in_categories')

error_prc_list = []
for error_number in error_number_list:
    error_prc_list.append(error_number * 64 / 100)

bar_chart(error_number_list, CATEGORIES, 'number_of_errors_in_categories')
bar_chart(error_prc_list, CATEGORIES, 'percent_of_errors_in_categories')

for prediction, image, fact in zip(predictions, X_test, y_test):
    if np.argmax(prediction) == fact:
        image_chart_combo(prediction, fact, CATEGORIES, image, f'prediction_{prediction_nr}')
    elif np.argmax(prediction) != fact:
        np.set_printoptions(suppress=True)  # suppress scientific print
        print(np.max(prediction))
        print(f"Error in category {fact}")
        error_count += 1
        image_chart_combo(prediction, fact, CATEGORIES, image, f'error_{prediction_nr}')
    prediction_nr += 1

print("# Predict accuracy")
print(100 - error_count / len(y_test) * 100)

print("-----------------------------------------------------------------------------")
print('# Evaluate on test data')
results = model.evaluate(X_test, y_test)
print('test loss, test acc:', results)
