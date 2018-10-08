import pickle

import bow_model
import utils
import settings
import torch_data_loader


def main(prep="prep_0"):
    """
    """
    print("Loading data...")
    try:
        # Load preprocessed data
        train = pickle.load(
            open(settings.DATA_DIR + "train.{}.pkl".format(prep), "rb"))
        train_toks = pickle.load(
            open(settings.DATA_DIR + "train.{}.toks.pkl".format(prep), "rb"))
        val = pickle.load(
            open(settings.DATA_DIR + "val.{}.pkl".format(prep), "rb"))
        val_toks = pickle.load(
            open(settings.DATA_DIR + "val.{}.toks.pkl".format(prep), "rb"))
    except Exception:
        # Preprocess data
        train, train_toks, val, val_toks = utils.preprocess_dataset("prep_0")

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
    EMD_DIM = 100
    model = bow_model.BOW(len(id2token), EMD_DIM)

    # Train
    bow_model.train(model, train_loader, val_loader)


if __name__ == "__main__":
    main()

