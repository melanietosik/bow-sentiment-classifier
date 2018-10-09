import pickle

import bow_model
import utils
import settings
import torch_data_loader


def trial(scheme):
    """
    Run trial
    """
    try:
        # Load preprocessed data
        print("Loading data...")
        train = pickle.load(
            open(settings.DATA_DIR + "train.{}.pkl".format(scheme), "rb"))
        train_toks = pickle.load(
            open(settings.DATA_DIR + "train.{}.toks.pkl".format(scheme), "rb"))
        val = pickle.load(
            open(settings.DATA_DIR + "val.{}.pkl".format(scheme), "rb"))
        val_toks = pickle.load(
            open(settings.DATA_DIR + "val.{}.toks.pkl".format(scheme), "rb"))
    except Exception:
        # Preprocess data
        print("Data not found, preprocessing...")
        train, train_toks, val, val_toks = utils.preprocess_dataset(scheme)

    # Split data samples and targets
    train_samples, train_targets = zip(*train)
    val_samples, val_targets = zip(*val)

    # Build vocab
    print("Building vocabulary...")
    token2id, id2token = utils.build_vocab(train_toks)

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
    model = bow_model.BOW(len(id2token))

    # Train
    train_acc, val_acc = bow_model.train(
        model, train_loader, val_loader)

    return train_acc, val_acc


def main():
    """
    Ablation study
    """
    # Tokenization schemes
    schemes = [0, 1, 2, 3]
    tokenization = {}
    for scheme in schemes:
        train_acc, val_acc = trial(scheme)
        tokenization[scheme] = {
            "train": train_acc,
            "val": val_acc,
        }
    print(tokenization)
    pickle.dump(tokenization, open("results/tokenization.pkl", "wb"))


if __name__ == "__main__":
    main()
