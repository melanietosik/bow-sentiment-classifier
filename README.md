# Bag-of-words (BOW) sentiment classifier

## Requirements

```
$ #(HPC) module load anaconda3/4.3.1
$ conda create -n bow python=3.6
$ source activate bow
$ conda install -c conda-forge spacy
$ python -m spacy download en
$ conda install pytorch torchvision -c pytorch
$ #(HPC) pip install torch torchvision
```

# module load anaconda3/4.3.1 cuda/9.0.176 cudnn/9.0v7.0.5
# du -h --max-depth=0 * | sort -hr

## Data

Download ["Large Movie Review Dataset v1.0"](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz) from [http://ai.stanford.edu/~amaas/data/sentiment/](http://ai.stanford.edu/~amaas/data/sentiment/)

> This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. We provide a set of 25,000 highly polar movie reviews for training, and 25,000 for testing. There is additional unlabeled data for use as well. Raw text and already processed bag of words formats are provided. See the README file contained in the release for more details.

```
$ mkdir data
$ cd data/
$ wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
$ tar -xvzf aclImdb_v1.tar.gz
$ #rm aclImdb_v1.tar.gz
```

## Ablation study

### Baseline

- Tokenization: `string.split()`
- Vocabulary size: `10,000`
- Embedding size: `100`
- N-gram size: `1`
- Optimizer: `Adam`
- Learning rate: `0.01`
- Number of epochs: `10`
- Dropout: `n/a`
- [fixed] Maximum sentence length: `200`
- [fixed] Batch size: `32`


### Tokenization

1. Tokenization using spaCy
2. Enhanced preprocessing using spaCy tokenization and filtering of stop words and punctuation
3. Enhanced preprocessing using spaCy tokenization, filtering of stop words and punctuation, and lemmatization

