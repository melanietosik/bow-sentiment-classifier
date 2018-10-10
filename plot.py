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
    ax.set_ylim(60, 100)
    lin = np.linspace(0, e, e * 6)
    # Scheme 0
    plt.plot(lin, r[0]["train"], color=CB[0], label="[0] train")
    plt.plot(lin, r[0]["val"], color=CB[0], linestyle="--", label="[0] val")
    # Scheme 1
    plt.plot(lin, r[1]["train"], color=CB[1], label="[1] train")
    plt.plot(lin, r[1]["val"], color=CB[1], linestyle="--", label="[1] val")
    # Scheme 2
    plt.plot(lin, r[2]["train"] , color=CB[2], label="[2] train")
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
    for key in r:
        print("scheme=[{}]; best training accuracy: {}; best validation accuracy: {}".format(
            key, max(r[key]["train"]), max(r[key]["val"])))


def adam_lr(e=10):
    r = pickle.load(open(r_dir + "adam_lr.pkl", "rb"))
    fig, ax = plt.subplots(1, 1)
    ax.set_ylim(60, 100)
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
    plt.savefig("plots/adam_lr.eps", format="eps", dpi=1000)
    for key in r:
        print("lr=[{}]; best training accuracy: {}; best validation accuracy: {}".format(
            key, max(r[key]["train"]), max(r[key]["val"])))


def ngrams(e=2, scheme=2, vocab_size=100000):
    r = pickle.load(
        open(r_dir + "ngrams.scheme={}.vocab={}.pkl".format(scheme, vocab_size), "rb"))
    fig, ax = plt.subplots(1, 1)
    ax.set_ylim(60, 100)
    lin = np.linspace(0, e, e * 6)
    # Scheme 0
    plt.plot(lin, r[1]["train"], color=CB[0], label="[n<=1] train")
    plt.plot(lin, r[1]["val"], color=CB[0], linestyle="--", label="[n<=1] val")
    # Scheme 1
    plt.plot(lin, r[2]["train"], color=CB[1], label="[n<=2] train")
    plt.plot(lin, r[2]["val"], color=CB[1], linestyle="--", label="[n<=2] val")
    # Scheme 2
    plt.plot(lin, r[3]["train"], color=CB[2], label="[n<=3] train")
    plt.plot(lin, r[3]["val"], color=CB[2], linestyle="--", label="[n<=3] val")
    # Scheme 3
    plt.plot(lin, r[4]["train"], color=CB[3], label="[n<=4] train")
    plt.plot(lin, r[4]["val"], color=CB[3], linestyle="--", label="[n<=4] val")
    ax.set_title("N-gram size [n<=1-4; vocab={}; scheme={}]".format(vocab_size, scheme))
    ax.set_xlabel("# of epochs")
    ax.set_ylabel("train/validation accuracy")
    ax.set_xticks(np.arange(0, e + 1, 1))
    ax.legend()
    plt.savefig("plots/ngrams.scheme={}.vocab={}.eps".format(scheme, vocab_size), format="eps", dpi=1000)
    for key in r:
        print("ngrams=[{}]; best training accuracy: {}; best validation accuracy: {}".format(
            key, max(r[key]["train"]), max(r[key]["val"])))


def emb_dims(e=2):
    r = pickle.load(
        open(r_dir + "emb_dims.pkl", "rb"))
    fig, ax = plt.subplots(1, 1)
    ax.set_ylim(60, 100)
    lin = np.linspace(0, e, e * 6)
    # 50
    plt.plot(lin, r[50]["train"], color=CB[0], label="[50] train")
    plt.plot(lin, r[50]["val"], color=CB[0], linestyle="--", label="[50] val")
    # 100
    plt.plot(lin, r[100]["train"], color=CB[1], label="[100] train")
    plt.plot(lin, r[100]["val"], color=CB[1], linestyle="--", label="[100] val")
    # 200
    plt.plot(lin, r[200]["train"], color=CB[2], label="[200] train")
    plt.plot(lin, r[200]["val"], color=CB[2], linestyle="--", label="[200] val")
    ax.set_title("Embedding dimensions [50, 100, 200]")
    ax.set_xlabel("# of epochs")
    ax.set_ylabel("train/validation accuracy")
    ax.set_xticks(np.arange(0, e + 1, 1))
    ax.legend()
    plt.savefig("plots/emb_dims.eps", format="eps", dpi=1000)
    for key in r:
        print("dim=[{}]; best training accuracy: {}; best validation accuracy: {}".format(
            key, max(r[key]["train"]), max(r[key]["val"])))


if __name__ == "__main__":
    #tokenization()
    """
        scheme=[0]; best training accuracy: 99.52; best validation accuracy: 86.0
        scheme=[1]; best training accuracy: 99.29; best validation accuracy: 85.88
        scheme=[2]; best training accuracy: 99.61; best validation accuracy: 87.94
        scheme=[3]; best training accuracy: 99.57; best validation accuracy: 87.42
    """

    #adam_lr()
    """
        lr=[0.01];  best training accuracy: 99.57;  best validation accuracy: 88.04
        lr=[0.001]; best training accuracy: 98.04;  best validation accuracy: 87.76
        lr=[0.0001]; best training accuracy: 85.2;  best validation accuracy: 84.16
    """

    #ngrams(scheme=2, vocab_size=10000)
    """
        ngrams=[1]; best training accuracy: 89.335; best validation accuracy: 87.2
        ngrams=[2]; best training accuracy: 88.52;  best validation accuracy: 85.6
        ngrams=[3]; best training accuracy: 87.315; best validation accuracy: 85.24
        ngrams=[4]; best training accuracy: 86.91;  best validation accuracy: 85.08
    """

    #ngrams(scheme=2, vocab_size=50000)
    """
        ngrams=[1]; best training accuracy: 91.53;  best validation accuracy: 87.88
        ngrams=[2]; best training accuracy: 90.57;  best validation accuracy: 85.02
        ngrams=[3]; best training accuracy: 89.925; best validation accuracy: 85.78
        ngrams=[4]; best training accuracy: 89.41;  best validation accuracy: 86.2
    """

    #ngrams(scheme=2, vocab_size=100000)
    """
        ngrams=[1]; best training accuracy: 91.45;  best validation accuracy: 87.48
        ngrams=[2]; best training accuracy: 91.785; best validation accuracy: 86.3
        ngrams=[3]; best training accuracy: 90.72;  best validation accuracy: 85.96
        ngrams=[4]; best training accuracy: 90.055; best validation accuracy: 86.32
    """

    #ngrams(scheme=1, vocab_size=10000, e=5)
    """
        ngrams=[1]; best training accuracy: 91.455; best validation accuracy: 86.34
        ngrams=[2]; best training accuracy: 89.205; best validation accuracy: 85.02
        ngrams=[3]; best training accuracy: 88.85;  best validation accuracy: 85.06
        ngrams=[4]; best training accuracy: 88.7;   best validation accuracy: 84.34
    """

    #ngrams(scheme=1, vocab_size=50000, e=5)
    """
        ngrams=[1]; best training accuracy: 95.09;  best validation accuracy: 87.44
        ngrams=[2]; best training accuracy: 94.415; best validation accuracy: 86.62
        ngrams=[3]; best training accuracy: 93.47;  best validation accuracy: 86.66
        ngrams=[4]; best training accuracy: 93.35;  best validation accuracy: 85.98
    """

    #ngrams(scheme=1, vocab_size=100000, e=5)
    """
        ngrams=[1]; best training accuracy: 95.855; best validation accuracy: 87.16
        ngrams=[2]; best training accuracy: 95.88;  best validation accuracy: 86.72
        ngrams=[3]; best training accuracy: 94.79;  best validation accuracy: 87.28
        ngrams=[4]; best training accuracy: 94.82;  best validation accuracy: 86.46
    """

    #emb_dims()
    """
        dim=[50]; best training accuracy: 89.615;   best validation accuracy: 86.48
        dim=[100]; best training accuracy: 91.525;  best validation accuracy: 87.36
        dim=[200]; best training accuracy: 93.1;    best validation accuracy: 88.14
    """

