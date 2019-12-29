import numpy as np
import cv2
from matplotlib import pyplot as plt

PLOTDIR = '../plots/'


def bar_chart(data, labels, title):
    plt.figure(figsize=(10, 8))
    plt.bar(labels, data, align='center')
    plt.ylabel('Ilość')
    plt.xticks(rotation=90)
    # plt.title(title)
    plt.subplots_adjust(bottom=0.4)
    plt.savefig(f'{PLOTDIR}{title}.png')


def image_mosaic(data, title, colormap):
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
    plt.savefig(f'{PLOTDIR}{title}.png')
    fig.clear()
