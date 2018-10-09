TRAIN_POS = "data/aclImdb/train/pos"
TRAIN_NEG = "data/aclImdb/train/neg"

TEST_POS = "data/aclImdb/test/pos"
TEST_NEG = "data/aclImdb/test/neg"

DATA_DIR = "data/"
RESULTS_DIR = "results/"

NUM_CLASSES = 2

CONFIG = {
    "scheme": 2,
    "num_epochs": 10,
    "ngram_size": 1,
    "max_vocab_size": 10000,
    "emb_dim": 100,
    "lr": -1,
    "max_sent_len": 200,
    "batch_size": 32,
}
