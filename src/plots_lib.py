import numpy as np
from matplotlib import pyplot as plt

PLOTDIR = '../plots/'


def bar_chart(labels, data, title):
    plt.bar(data, labels, align='center')
    plt.ylabel('Ilość')
    plt.xticks(rotation=90)
    plt.title(title)
    plt.subplots_adjust(bottom=0.4)
    plt.savefig(f'{PLOTDIR}{title}.png')


def image_mosaic(data, title):
    fig = plt.figure(figsize=(6, 4))
    fig.suptitle(title)
    columns = 8
    rows = 8
    for i in range(1, columns * rows + 1):
        img = data[i - 1]
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        plt.imshow(np.squeeze(img), cmap='gray')
    plt.savefig(f'{PLOTDIR}{title}.png')
