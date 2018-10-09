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

# Tokenization and epochs
for e in [1, 5, 10]:
    tok_r = pickle.load(open(r_dir + "tokenization.epochs={}.pkl".format(e), "rb"))
    fig, ax = plt.subplots(1, 1)
    lin = np.linspace(0, e, e * 6)
    # Scheme 0
    plt.plot(lin, tok_r[0]["train"], color=CB[0], label="[0] train")
    plt.plot(lin, tok_r[0]["val"], color=CB[0], linestyle="--", label="[0] val")
    # Scheme 1
    plt.plot(lin, tok_r[1]["train"], color=CB[1], label="[1] train")
    plt.plot(lin, tok_r[1]["val"], color=CB[1], linestyle="--", label="[1] val")
    # Scheme 2
    plt.plot(lin, tok_r[2]["train"], color=CB[2], label="[2] train")
    plt.plot(lin, tok_r[2]["val"], color=CB[2], linestyle="--", label="[2] val")
    # Scheme 3
    plt.plot(lin, tok_r[3]["train"], color=CB[3], label="[3] train")
    plt.plot(lin, tok_r[3]["val"], color=CB[3], linestyle="--", label="[3] val")
    ax.set_title("Tokenization schemes [0-3]")
    ax.set_xlabel("# of epochs")
    ax.set_ylabel("train/validation accuracy")
    ax.set_xticks(np.arange(0, e + 1, 1))
    ax.legend()
    plt.savefig("plots/tokenization.epochs={}.eps".format(e), format="eps", dpi=1000)


