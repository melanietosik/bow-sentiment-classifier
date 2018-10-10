# Bag-of-words sentiment classifier

[Natural Language Processing with Representation Learning (DS-GA 1011)](https://docs.google.com/document/d/1o0TTWocbkqPa9qsTCXnEFXf3NZzwZLLLSw7SSZmNla8/edit#)

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

## Overview

See `main.py` for the main script to run. Change default parameter settings in `settings.py`. Slightly modified versions of the lab code can be found in `bow_model.py` and `torch_data_loader.py`. The `utils.py` script contains all of the preprocessing code. Finally, the plots and result tables are generated in `plot.py` (see also `plots/` and `results/`).

## Preliminary results

### Ablation study

- Tokenzation schemes
- Number of epochs
- N-gram size
- Vocabulary size
- Embedding size
- Optimimizer (Adam vs. SGD)
- Learning rate

#### Tokenization

0. Baseline: `string.split()`
1. Tokenization using spaCy
2. Tokenization using spaCy, filtering of stop words and punctuation
3. Tokenization using spaCy, filtering of stop words and punctuation, lemmatization

So far, the second tokenization scheme `[2]` works best. Lemmatization seems to be overkill, but filtering stop words and punctuation is helpful. It looks like the model is overfitting though, so let's adjust the learning rate next.

####  Learning rate

The default learning rate was set to `0.01`, which is pretty high for the Adam optimizer. Results:
	
- `1e-2`: clearly overfitting
- `1e-3`: nice learning curve
- `1e-4`: too slow

We will stick with a learning of `1e-3` for Adam for now. It looks like we can also reduce the number of epochs from 10 to 2 for the following experiments.

####  N-gram and vocabulary size

Testing n-gram sizes between `[1-4]` surprisingly did not yield very promising results. Evaluating different vocabulary sizes [`10k, 50k, 100k`] did not significantly affect model performance either. Reverting the tokenization scheme back to `[1]` and _including_ stop words and punctuation improves the results, especially when using bigrams over unigrams. Increasing the number of epochs to 5 again was necessary for the learning curve to converge. Still, the best results are consistently achieved with the more rigorous tokenization scheme `[2]` and using unigrams [`n=1`] with a maximum vocabulary size of `[50k`].

#### Embedding size

Results: `200d` > `100d` > `50d`.

#### Optimizer

We will compare `Adam` vs. `SGD`, both with default parameters. SGD doesn't seem to be working well at all. We will keep using `Adam` for now.

#### Linear annealing of learning rate

Not helpful.

#### Number of epochs





