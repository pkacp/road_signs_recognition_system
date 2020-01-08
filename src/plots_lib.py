import numpy as np
import cv2
from matplotlib import pyplot as plt
from settings import PLOTDIR


def bar_chart(data, labels, title):
    plt.figure(figsize=(10, 8))
    plt.bar(labels, data, align='center')
    plt.ylabel('Ilość')
    plt.xticks(rotation=90)
    # plt.title(title)
    plt.subplots_adjust(bottom=0.4)
    plt.savefig(f'{PLOTDIR}/{title}.png')


def double_bar_chart(data1, data1_label, data2, data2_label, labels, title):
    x = np.arange(len(labels))
    width = 0.4
    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot()
    ax.bar(x - width / 2, data1, width, label=data1_label)
    ax.bar(x + width / 2, data2, width, label=data2_label)
    ax.set_ylabel('Ilość')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.4)
    plt.savefig(f'{PLOTDIR}/{title}.png')
    fig.clear()


def image_mosaic(data, title):
    channels = data[0][0][0].shape[0]
    if channels == 3:
        colormap = 'rgb'
    elif channels == 1:
        colormap = 'gray'
    plt.style.use('default')
    fig = plt.figure(figsize=(10, 8))
    # fig.suptitle(title)
    columns = 8
    rows = 8
    for i in range(1, columns * rows + 1):
        img = data[i - 1]
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        if colormap == 'gray':
            plt.imshow(np.squeeze(img), cmap='gray')
        elif colormap == 'rgb':
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(np.squeeze(img))
    plt.savefig(f'{PLOTDIR}/{colormap}_{title}.png')
    fig.clear()


def plot_multi_val_accuracy(history_arr, labels_arr, acc_method, title):
    plt.clf()
    for history in history_arr:
        plt.plot(map(lambda x: x * 100, history.history[f'val_{acc_method}']))
    plt.ylabel('Dokładność [%]')
    plt.xlabel('Epoka uczenia')
    plt.legend(labels_arr, loc='upper left')
    plt.savefig(f'{PLOTDIR}/{title}.png')


def plot_multi_val_loss(history_arr, labels_arr, title):
    plt.clf()
    for history in history_arr:
        plt.plot(history.history['val_loss'])
    plt.ylabel('Strata')
    plt.xlabel('Epoka uczenia')
    plt.legend(labels_arr, loc='upper left')
    plt.savefig(f'{PLOTDIR}/{title}.png')


def plot_accuracy_history(history, acc_method, title):
    print("AAAAAAAAAAAAAAAAAAAAA")
    plt.clf()
    plt.plot(map(lambda x: x * 100, history.history[acc_method]))
    plt.plot(map(lambda x: x * 100, history.history[f'val_{acc_method}']))
    plt.ylabel('Dokładność [%]')
    plt.xlabel('Epoka uczenia')
    plt.legend(['Zbiór uczący', 'Zbiór walidacyjny'], loc='upper left')
    plt.savefig(f'{PLOTDIR}/{title}.png')


def plot_loss_history(history, loss_method, title):
    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('Strata')
    plt.xlabel('Epoka uczenia')
    plt.legend(['Zbiór uczący', 'Zbiór walidacyjny'], loc='upper left')
    plt.savefig(f'{PLOTDIR}/{title}.png')


def plot_image(predictions_array, true_label, categories_array, img):
    img = img * 255.0
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(np.squeeze(img), cmap='gray')

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel(
        f"{categories_array[predicted_label]} {100 * np.max(predictions_array)}% ({categories_array[true_label]})",
        color=color)


def plot_value_array(predictions_array, true_label, categories_array):
    # predictions_array, true_label = predictions_array, true_label[i]
    categories_count = len(categories_array)
    plt.grid(False)
    plt.xticks(range(categories_count))
    plt.yticks([])
    thisplot = plt.bar(range(categories_count), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def image_chart_combo(predictions_array, true_label, categories_array, img):
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(predictions_array, true_label, categories_array, img)
    plt.subplot(1, 2, 2)
    plot_value_array(predictions_array, true_label, categories_array)
    # plt.savefig(f'{PLOTDIR}/invalid_prediction.png')
