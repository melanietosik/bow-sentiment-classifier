import pickle

import bow_model
import utils
import settings
import torch_data_loader


def trial(
    scheme=settings.CONFIG["scheme"],
    n=settings.CONFIG["ngram_size"],
    lr=settings.CONFIG["lr"],
    vocab_size=settings.CONFIG["max_vocab_size"],
    dim=settings.CONFIG["emb_dim"],
    optim=settings.CONFIG["optim"],
    lin_ann=settings.CONFIG["lin_ann"],
):
    """
    Run trial
    """
    try:
        # Load preprocessed data
        print("Loading data...")
        train = pickle.load(open(
            settings.DATA_DIR + "train.{}.n={}.pkl".format(scheme, n), "rb"))
        train_toks = pickle.load(open(
            settings.DATA_DIR + "train.{}.n={}.toks.pkl".format(scheme, n), "rb"))
        val = pickle.load(open(
            settings.DATA_DIR + "val.{}.n={}.pkl".format(scheme, n), "rb"))
        val_toks = pickle.load(open(
            settings.DATA_DIR + "val.{}.n={}.toks.pkl".format(scheme, n), "rb"))
    except Exception:
        # Preprocess data
        print("Data not found, preprocessing...")
        train, train_toks, val, val_toks = utils.preprocess_dataset(scheme, n)

    # Split data samples and targets
    train_samples, train_targets = zip(*train)
    val_samples, val_targets = zip(*val)

    # Build vocab
    print("Building vocabulary...")
    token2id, id2token = utils.build_vocab(train_toks, vocab_size)

    # Convert tokens to IDs
    print("Converting tokens to indices...")
    train_idxs = utils.tok2idx_data(token2id, train_samples)
    val_idxs = utils.tok2idx_data(token2id, val_samples)
    assert(len(train_idxs) == len(train_samples))
    assert(len(val_idxs) == len(val_samples))

    # PyTorch data loader
    print("Creating PyTorch data loaders...")
    train_loader = torch_data_loader.get(
        train_idxs, train_targets, shuffle=True)
    val_loader = torch_data_loader.get(
        val_idxs, val_targets, shuffle=True)

    # BOW model
    print("Building BOW model...")
    model = bow_model.BOW(len(id2token), dim)

    # Train
    train_acc, val_acc = bow_model.train(
        model,
        train_loader,
        val_loader,
        lr,
        optim,
        lin_ann,
    )
    return train_acc, val_acc


def main():
    """
    Ablation study
    """
    # # Tokenization schemes
    # schemes = [0, 1, 2, 3]
    # tokenization = {}
    # for scheme in schemes:
    #     train_acc, val_acc = trial(scheme)
    #     tokenization[scheme] = {
    #         "train": train_acc,
    #         "val": val_acc,
    #     }
    # print(tokenization)
    # pickle.dump(tokenization, open("results/tokenization.pkl", "wb"))

    # # Learning rate (Adam; default: 1e-3)
    # lr = [1e-2, 1e-3, 1e-4]
    # adam_lr = {}
    # for rate in lr:
    #     train_acc, val_acc = trial(lr=rate)
    #     adam_lr[rate] = {
    #         "train": train_acc,
    #         "val": val_acc,
    #     }
    # print(adam_lr)
    # pickle.dump(adam_lr, open("results/adam_lr.pkl", "wb"))

    # # N-gram size
    # size = [1, 2, 3, 4]
    # vocab_size = 10000
    # ngrams = {}
    # for n in size:
    #     train_acc, val_acc = trial(scheme=1, n=n, vocab_size=vocab_size)
    #     ngrams[n] = {
    #         "train": train_acc,
    #         "val": val_acc,
    #     }
    # print(ngrams)
    # pickle.dump(
    #     ngrams,
    #     open("results/ngrams.scheme=1.vocab={}.pkl".format(vocab_size), "wb"))

    # # Embedding size
    # dims = [50, 100, 200]
    # emb_dims = {}
    # for dim in dims:
    #     print("dim:", dim)
    #     train_acc, val_acc = trial(dim=dim)
    #     emb_dims[dim] = {
    #         "train": train_acc,
    #         "val": val_acc,
    #     }
    # print(emb_dims)
    # pickle.dump(emb_dims, open("results/emb_dims.pkl", "wb"))

    # # Optimizer
    # defaults = {
    #     "adam": 1e-3,
    #     "sgd": 1e-2,
    # }
    # optims = {}
    # for optim in defaults:
    #     print("optim:", optim)
    #     train_acc, val_acc = trial(optim=optim, lr=defaults[optim])
    #     optims[optim] = {
    #         "train": train_acc,
    #         "val": val_acc,
    #     }
    # print(optims)
    # pickle.dump(optims, open("results/optims.pkl", "wb"))

    # Linear annealing of learning rate
    options = [True, False]
    for boolean in options:
        print("lin_ann:", boolean)
        train_acc, val_acc = trial(lin_ann=boolean)
        results[str(boolean)] = {
            "train": train_acc,
            "val": val_acc,
        }
    print(results)
    pickle.dump(results, open("results/annealing.pkl", "wb"))


if __name__ == "__main__":
    main()
