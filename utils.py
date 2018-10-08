import os
import pickle
import string
import random

import spacy

import settings

nlp = spacy.load(
    "en",
    # disable=[
    #     "tagger,"
    #     "parser",
    # ]
)
# nlp.add_pipe(nlp.create_pipe("sentencizer"))


def preprocess(text):
    return text.split()


def preprocess_dataset():
    """
    """
    print("Preprocessing dataset...")
    data = []
    # test = []

    # Training and validation set
    for label, dir_ in enumerate([settings.TRAIN_POS, settings.TRAIN_NEG]):
        for file in os.listdir(dir_):
            if file.endswith(".txt"):
                text = open(os.path.join(dir_, file), "r").read()
                data.append((preprocess(text), label))

    # Shuffle training data
    random.shuffle(data)
    train = data[:20000]
    val = data[20000:]

    print("# train samples:", len(train))
    print("# val samples:", len(val))

    # for label, dir_ in enumerate([settings.TRAIN_POS, settings.TRAIN_NEG]):
    # for file in os.listdir(dir_):
    #     if file.endswith(".txt"):
    #         text = open(os.path.join(dir_, file), "r").read()
    #         data.append((preprocess(text), label))

    pickle.dump(train, open(settings.DATA_DIR + "train_prep_0.pkl", "wb"))
    pickle.dump(val, open(settings.DATA_DIR + "val_prep_0.pkl", "wb"))
    print("Done")
