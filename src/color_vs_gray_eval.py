import cv2
import pickle
import tensorflow as tf
import numpy as np
from settings import *
from plots_lib import *

# read rgb
X_val_rgb = np.array(pickle.load(open('../pickled_datasets/rgb/X_val.pickle', "rb")))
y_val_rgb = np.array(pickle.load(open('../pickled_datasets/rgb/y_val.pickle', "rb")))

X_test_rgb = np.array(pickle.load(open('../pickled_datasets/rgb/X_test.pickle', "rb")))
y_test_rgb = np.array(pickle.load(open('../pickled_datasets/rgb/y_test.pickle', "rb")))

X_val_rgb = X_val_rgb / 255.0
X_test_rgb = X_test_rgb / 255.0

model_name_rgb = '1578714434-rgb-road_signs_recognition-conv-32x32x64x128-sigmoid.model'
model_rgb = tf.keras.models.load_model(f'../saved_models/{model_name_rgb}')

# read bw
X_val_bw = np.array(pickle.load(open('../pickled_datasets/gray/X_val.pickle', "rb")))
y_val_bw = np.array(pickle.load(open('../pickled_datasets/gray/y_val.pickle', "rb")))

X_test_bw = np.array(pickle.load(open('../pickled_datasets/gray/X_test.pickle', "rb")))
y_test_bw = np.array(pickle.load(open('../pickled_datasets/gray/y_test.pickle', "rb")))

X_val_bw = X_val_bw / 255.0
X_test_bw = X_test_bw / 255.0

model_name_bw = '1578721238-gray-road_signs_recognition-conv-32x32x64x128-sigmoid.model'
model_bw = tf.keras.models.load_model(f'../saved_models/{model_name_bw}')

predictions_rgb = model_rgb.predict(X_test_rgb)
predictions_bw = model_bw.predict(X_test_bw)

# calculations for rgb
error_number_list_rgb = [0] * len(CATEGORIES)
all_correct_prediction_procent_rgb = [[] for i in range(len(CATEGORIES))]
all_incorrect_prediction_procent_rgb = [[] for i in range(len(CATEGORIES))]
print(all_incorrect_prediction_procent_rgb)

for prediction, fact in zip(predictions_rgb, y_test_rgb):
    id_max_prediction_confidence = np.argmax(prediction)
    max_prediction_confidence = np.max(prediction)
    if id_max_prediction_confidence != fact:
        error_number_list_rgb[fact] += 1
        all_incorrect_prediction_procent_rgb[fact].append(max_prediction_confidence)
    elif id_max_prediction_confidence == fact:
        all_correct_prediction_procent_rgb[fact].append(max_prediction_confidence)

mean_all_correct_prediction_procent_rgb = [0] * len(CATEGORIES)
for idx, category_correct_procent_predictions in enumerate(all_correct_prediction_procent_rgb):
    mean_all_correct_prediction_procent_rgb[idx] = sum(category_correct_procent_predictions) / len(category_correct_procent_predictions)

mean_all_incorrect_prediction_procent_rgb = [0] * len(CATEGORIES)
print(all_incorrect_prediction_procent_rgb)
for idx, category_incorrect_procent_predictions in enumerate(all_incorrect_prediction_procent_rgb):
    if len(category_incorrect_procent_predictions) == 0:
        mean_all_incorrect_prediction_procent_rgb[idx] = 0.0
    else:
        mean_all_incorrect_prediction_procent_rgb[idx] = sum(category_incorrect_procent_predictions) / len(category_incorrect_procent_predictions)

error_prc_list_rgb = []
for error_number in error_number_list_rgb:
    error_prc_list_rgb.append(error_number * 64 / 100)

# calculations for bw
error_number_list_bw = [0] * len(CATEGORIES)
all_correct_prediction_procent_bw = [[] for i in range(len(CATEGORIES))]
all_incorrect_prediction_procent_bw = [[] for i in range(len(CATEGORIES))]

for prediction, fact in zip(predictions_bw, y_test_bw):
    id_max_prediction_confidence = np.argmax(prediction)
    max_prediction_confidence = np.max(prediction)
    if id_max_prediction_confidence != fact:
        error_number_list_bw[fact] += 1
        all_incorrect_prediction_procent_bw[fact].append(max_prediction_confidence)
    elif id_max_prediction_confidence == fact:
        all_correct_prediction_procent_bw[fact].append(max_prediction_confidence)

mean_all_correct_prediction_procent_bw = [0] * len(CATEGORIES)
for idx, category_correct_procent_predictions in enumerate(all_correct_prediction_procent_bw):
    mean_all_correct_prediction_procent_bw[idx] = sum(category_correct_procent_predictions) / len(category_correct_procent_predictions)

mean_all_incorrect_prediction_procent_bw = [0] * len(CATEGORIES)
for idx, category_incorrect_procent_predictions in enumerate(all_incorrect_prediction_procent_bw):
    if len(category_incorrect_procent_predictions) == 0:
        mean_all_incorrect_prediction_procent_bw[idx] = 0.0
    else:
        mean_all_incorrect_prediction_procent_bw[idx] = sum(category_incorrect_procent_predictions) / len(category_incorrect_procent_predictions)

error_prc_list_bw = []
for error_number in error_number_list_bw:
    error_prc_list_bw.append(error_number * 64 / 100)

# draw plots
double_bar_chart([i * 100 for i in mean_all_correct_prediction_procent_rgb],
                 'Sieć operująca na zbiorze zdjęć kolorowych',
                 [i * 100 for i in mean_all_correct_prediction_procent_bw],
                 'Sieć operująca na zbiorze zdjęć w skali szarości',
                 CATEGORIES, 'mean_percent_correct_prediction_rgb_vs_bw', 'Kategoria znaku', 'Średni procent pewności dla poprawnych predykcji')
double_bar_chart([i * 100 for i in mean_all_incorrect_prediction_procent_rgb],
                 'Sieć operująca na zbiorze zdjęć kolorowych',
                 [i * 100 for i in mean_all_incorrect_prediction_procent_bw],
                 'Sieć operująca na zbiorze zdjęć w skali szarości',
                 CATEGORIES, 'mean_percent_incorrect_prediction_rgb_vs_bw', 'Kategoria znaku', 'Średni procent pewności dla niepoprawnych predykcji')
print('AAA')
double_bar_chart(error_prc_list_rgb, "Sieć operująca na zbiorze zdjęć kolorowych",
                 error_number_list_bw, "Sieć operująca na zbiorze zdjęć w skali szarości",
                 CATEGORIES, 'error_percent_prediction_rgb_vs_bw', 'Kategoria znaku', 'Procent błędnych rozpoznań w kategorii')
