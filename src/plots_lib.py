from matplotlib import pyplot as plt

PLOTDIR = '../plots/'


def bar_chart(labels, data, title):
    plt.bar(data, labels, align='center')
    plt.ylabel('Ilość')
    plt.xticks(rotation=90)
    plt.title(title)
    plt.subplots_adjust(bottom=0.4)
    plt.savefig(f'{PLOTDIR}{title}.png')




# bar_chart(["aaa","vvv", "ccc"], [10,15,121], "asjdfoisadiofjo")
