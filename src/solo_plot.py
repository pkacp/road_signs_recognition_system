from matplotlib import pyplot as plt
from settings import *


def plot_accuracy_history(train_acc, val_acc, acc_method, title):
    train_acc = [i * 100 for i in train_acc]
    val_acc = [i * 100 for i in val_acc]
    plt.clf()
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.ylabel('Dokładność (%)')
    plt.xlabel('Epoka uczenia')
    plt.legend(['Zbiór uczący', 'Zbiór walidacyjny'], loc='upper left')
    plt.savefig(f'{PLOTDIR}/{title}.png', bbox_inches='tight')


def plot_loss_history(loss, val_loss, loss_method, title):
    plt.clf()
    plt.plot(loss)
    plt.plot(val_loss)
    plt.ylabel('Strata')
    plt.xlabel('Epoka uczenia')
    plt.legend(['Zbiór uczący', 'Zbiór walidacyjny'], loc='upper left')
    plt.savefig(f'{PLOTDIR}/{title}.png', bbox_inches='tight')


# rgb
# train_acc = [0.8805, 0.9605, 0.9705, 0.9756, 0.9770, 0.9802]
# val_acc = [0.9398, 0.9570, 0.9567, 0.9620, 0.9627, 0.9656]
# plot_accuracy_history(train_acc, val_acc, '', 'acc_recover_best_rgb_net')
# loss = [0.3814, 0.1267, 0.0970, 0.0792, 0.0724, 0.0623]
# val_loss = [0.1972, 0.1460, 0.1445, 0.1266, 0.1202, 0.1242]
# plot_loss_history(loss, val_loss, '', 'loss_recover_best_rgb_net')

#bw
train_acc = [0.8805, 0.9605, 0.9705, 0.9756, 0.9770, 0.9802]
val_acc = [0.9398, 0.9570, 0.9567, 0.9620, 0.9627, 0.9656]
plot_accuracy_history(train_acc, val_acc, '', 'acc_recover_best_gray_net')
loss = [0.3814, 0.1267, 0.0970, 0.0792, 0.0724, 0.0623]
val_loss = [0.1972, 0.1460, 0.1445, 0.1266, 0.1202, 0.1242]
plot_loss_history(loss, val_loss, '', 'loss_recover_best_gray_net')
