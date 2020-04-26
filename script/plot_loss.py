from matplotlib import pyplot as plt
import pickle
import numpy as np
import sys

if __name__ == "__main__":
    with open(sys.argv[1], "rb") as f:
        loss_hist = np.asarray(pickle.load(f))
    print(loss_hist.shape)
    if len(sys.argv) > 3:
        end = int(sys.argv[3])
    else:
        end = loss_hist.shape[0]
    t = np.arange(1, end+1)
    point5 = np.empty_like(t, dtype=np.float32)
    point5.fill(0.5)
    fig, ax = plt.subplots()
    ax.plot(t, point5,'k:', label='50%')
    ax.plot(t, loss_hist[:end, 0], label='training loss')
    ax.plot(t, loss_hist[:end, 1], label='training accuracy')
    ax.plot(t, loss_hist[:end, 2], label='validation accuracy')
    ax.legend()
    ax.set(xlabel='epochs', ylabel='loss or accuracy',
        title='{} training samples'.format(sys.argv[2]))
    ax.grid()
    fig.savefig("loss_hist.png")
    plt.show()
