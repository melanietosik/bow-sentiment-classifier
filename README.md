# Bag-of-words (BOW) sentiment classifier

## Requirements

```
$ #(HPC) module load anaconda3/4.3.1
$ conda create -n bow python=3.6
$ source activate bow
$ conda install -c conda-forge spacy
$ python -m spacy download en
$ #(loc) conda install pytorch torchvision -c pytorch
$ #(HPC) pip install torch torchvision
```

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

- Tokenzation schemes
- Number of epochs
- N-gram size
- Vocabulary size
- Embedding size
- Optimimizer
- Learning rate
- Dropout

### Tokenization

0. Baseline (`string.split()`)
1. Tokenization using spaCy
2. Tokenization using spaCy, filtering of stop words and punctuation
3. Tokenization using spaCy, filtering of stop words and punctuation, lemmatization

Evidently, the second tokenization scheme [2] works best. Lemmatization seems to be overkill, but filtering stop words and punctuation is helpful. It looks like the model is overfitting though, so let's adjust the learning rate next.

### Learning rate

The default learning rate was set to 0.01, which is pretty high for the Adam optimizer. Evaluating:
	
	- 1e-2: clearly overfitting
	- 1e-3: pretty good
	- 1e-4: too slow

Going to stick with 1e-3 and reduce the number of epochs from 10 to 2.

### N-gram size





