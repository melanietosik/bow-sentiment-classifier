import numpy as np
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt

import settings

mpl.rc("lines", linewidth=1)
CB = ["#377eb8", "#ff7f00", "#4daf4a",
      "#f781bf", "#a65628", "#984ea3",
      "#999999", "#e41a1c", "#dede00"]

r_dir = settings.RESULTS_DIR


def tokenization(e=10):
    r = pickle.load(open(r_dir + "tokenization.epochs={}.pkl".format(e), "rb"))
    fig, ax = plt.subplots(1, 1)
    lin = np.linspace(0, e, e * 6)
    # Scheme 0
    plt.plot(lin, r[0]["train"], color=CB[0], label="[0] train")
    plt.plot(lin, r[0]["val"], color=CB[0], linestyle="--", label="[0] val")
    # Scheme 1
    plt.plot(lin, r[1]["train"], color=CB[1], label="[1] train")
    plt.plot(lin, r[1]["val"], color=CB[1], linestyle="--", label="[1] val")
    # Scheme 2
    plt.plot(lin, r[2]["train"], color=CB[2], label="[2] train")
    plt.plot(lin, r[2]["val"], color=CB[2], linestyle="--", label="[2] val")
    # Scheme 3
    plt.plot(lin, r[3]["train"], color=CB[3], label="[3] train")
    plt.plot(lin, r[3]["val"], color=CB[3], linestyle="--", label="[3] val")
    ax.set_title("Tokenization schemes [0-3]")
    ax.set_xlabel("# of epochs")
    ax.set_ylabel("train/validation accuracy")
    ax.set_xticks(np.arange(0, e + 1, 1))
    ax.legend()
    plt.savefig("plots/tokenization.eps", format="eps", dpi=1000)


def adam_lr(e=10):
    r = pickle.load(open(r_dir + "adam_lr.pkl", "rb"))
    fig, ax = plt.subplots(1, 1)
    lin = np.linspace(0, e, e * 6)
    # 0.01
    plt.plot(lin, r[1e-2]["train"], color=CB[0], label="[1e-2] train")
    plt.plot(lin, r[1e-2]["val"], color=CB[0], linestyle="--", label="[1e-2] val")
    # 0.001
    plt.plot(lin, r[1e-3]["train"], color=CB[1], label="[1e-3] train")
    plt.plot(lin, r[1e-3]["val"], color=CB[1], linestyle="--", label="[1e-3] val")
    # 0.0001
    plt.plot(lin, r[1e-4]["train"], color=CB[3], label="[1e-4] train")
    plt.plot(lin, r[1e-4]["val"], color=CB[3], linestyle="--", label="[1e-4] val")
    ax.set_title("Adam learning rate [1e-2,3,4]")
    ax.set_xlabel("# of epochs")
    ax.set_ylabel("train/validation accuracy")
    ax.set_xticks(np.arange(0, e + 1, 1))
    ax.legend()
    plt.savefig("plots/adam_learning_rate.eps", format="eps", dpi=1000)


# # N-grams (vocab_size: 10k)
# e = 5
# ngrams_r = pickle.load(open(r_dir + "ngrams.pkl", "rb"))
# fig, ax = plt.subplots(1, 1)
# lin = np.linspace(0, e, e * 6)
# # Scheme 0
# plt.plot(lin, ngrams_r[1]["train"], color=CB[0], label="[1] train")
# plt.plot(lin, ngrams_r[1]["val"], color=CB[0], linestyle="--", label="[1] val")
# # Scheme 1
# plt.plot(lin, ngrams_r[2]["train"], color=CB[1], label="[2] train")
# plt.plot(lin, ngrams_r[2]["val"], color=CB[1], linestyle="--", label="[2] val")
# # Scheme 2
# plt.plot(lin, ngrams_r[3]["train"], color=CB[2], label="[3] train")
# plt.plot(lin, ngrams_r[3]["val"], color=CB[2], linestyle="--", label="[3] val")
# # Scheme 3
# plt.plot(lin, ngrams_r[4]["train"], color=CB[3], label="[4] train")
# plt.plot(lin, ngrams_r[4]["val"], color=CB[3], linestyle="--", label="[4] val")
# ax.set_title("N-gram size [1-4]")
# ax.set_xlabel("# of epochs")
# ax.set_ylabel("train/validation accuracy")
# ax.set_xticks(np.arange(0, e + 1, 1))
# ax.legend()
# plt.show()
#plt.savefig("plots/tokenization.epochs={}.eps".format(e), format="eps", dpi=1000)


if __name__ == "__main__":
    #tokenization()
    adam_lr()

