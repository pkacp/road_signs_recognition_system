import numpy as np
import inflection
from matplotlib import pyplot as plt

PLOTDIR = '../plots/'


def bar_chart(labels, data, title):
    # plt.xlabel('Months')
    # plt.ylabel('Books Read')
    # plt.show()
    plt.bar(data, labels, align='center')
    plt.ylabel('Ilość')
    plt.xticks(rotation=45)
    plt.title(title)
    plt.savefig(f'{PLOTDIR}{title}.png')


# bar_chart(["aaa","vvv", "ccc"], [10,15,121], "asjdfoisadiofjo")
