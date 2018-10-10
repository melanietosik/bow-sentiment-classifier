import os
import pickle
import random

import spacy

from collections import Counter
from itertools import chain
from spacy.lang.en.stop_words import STOP_WORDS
from tqdm import tqdm

import settings

nlp = spacy.load(
    "en",
    disable=[
        "tagger,"
        "parser",
        "ner",
    ]
)
nlp.add_pipe(nlp.create_pipe("sentencizer"))
random.seed(42)

PAD_IDX = 0
UNK_IDX = 1


def ngrams_(text, n):
    """
    Generate exactly n n-grams
    """
    return zip(*[text[i:] for i in range(n)])


def ngrams(text, n):
    """
    Generate up to n<=n n-grams
    """
    return chain(*[ngrams_(text, i) for i in range(n + 1)])


def preprocess(text, version, n):
    """
    Preprocessing
    """
    # Scheme 0
    if version == 0:
        prep = text.split()
        return ["_".join(ngram) for ngram in ngrams(prep, n)]
    # Scheme 1
    elif version == 1:
        doc = nlp(text)
        prep = [tok.text for tok in doc]
        return ["_".join(ngram) for ngram in ngrams(prep, n)]
    # Scheme 2
    elif version == 2:
        doc = nlp(text)
        prep = []
        for tok in doc:
            if tok.is_alpha:
                if (tok.lower_ in STOP_WORDS) or (tok.lemma_ in STOP_WORDS):
                    pass
                else:
                    prep.append(tok.lower_)
        return ["_".join(ngram) for ngram in ngrams(prep, n)]
    # Scheme 3
    elif version == 3:
        doc = nlp(text)
        prep = []
        for tok in doc:
            if tok.is_alpha:
                if (tok.lower_ in STOP_WORDS) or (tok.lemma_ in STOP_WORDS):
                    pass
                else:
                    prep.append(tok.lemma_)
        return ["_".join(ngram) for ngram in ngrams(prep, n)]
    else:
        print("Invalid tokenization scheme, exiting")
        exit()


def preprocess_dataset(version, n):
    """
    """
    data = []
    # test = []

    # Training and validation set
    for label, dir_ in enumerate([settings.TRAIN_POS, settings.TRAIN_NEG]):
        print(dir_)
        for file in tqdm(os.listdir(dir_)):
            if file.endswith(".txt"):
                text = open(os.path.join(dir_, file), "r").read()
                data.append((preprocess(text, version, n), label))

    # Shuffle training data
    random.shuffle(data)
    train = data[:20000]
    val = data[20000:]
    print("# train samples:", len(train))
    print("# val samples:", len(val))

    train_toks = []
    for text, _ in train:
        train_toks.extend(text)
    val_toks = []
    for text, _ in val:
        val_toks.extend(text)
    print("# train toks:", len(train_toks))
    print("# val toks:", len(val_toks))

    pickle.dump(train, open(
        settings.DATA_DIR + "train.{}.n={}.pkl".format(version, n), "wb"))
    pickle.dump(train_toks, open(
        settings.DATA_DIR + "train.{}.n={}.toks.pkl".format(version, n), "wb"))
    pickle.dump(val, open(
        settings.DATA_DIR + "val.{}.n={}.pkl".format(version, n), "wb"))
    pickle.dump(val_toks, open(
        settings.DATA_DIR + "val.{}.n={}.toks.pkl".format(version, n), "wb"))

    return train, train_toks, val, val_toks


def build_vocab(toks, max_vocab_size):
    """
    - id2token:
        list of tokens;
        id2token[i] returns token corresponding to token i
    - token2id:
        dictionary;
        keys represent tokens, corresponding values represent indices
    """
    ngram_cnt = Counter(toks)
    vocab, cnt = zip(*ngram_cnt.most_common(max_vocab_size))

    id2token = list(vocab)
    token2id = dict(zip(vocab, range(2, 2 + len(vocab))))

    id2token = ["<pad>", "<unk>"] + id2token
    token2id["<pad>"] = PAD_IDX
    token2id["<unk>"] = UNK_IDX

    # rand_tok_id = random.randint(0, len(id2token) - 1)
    # rand_tok = id2token[rand_tok_id]
    # print(rand_tok_id, id2token[rand_tok_id])
    # print(rand_tok, token2id[rand_tok])

    return token2id, id2token


def tok2idx_data(token2id, tok_data):
    """
    Convert tokens to IDs
    """
    idx_data = []
    for toks in tok_data:
        idx_lst = [
            token2id[tok] if tok in token2id else UNK_IDX for tok in toks]
        idx_data.append(idx_lst)
    return idx_data
