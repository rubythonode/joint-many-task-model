# A Joint Many-Task Model: Growing a Neural Network for Multiple NLP Tasks, ICLR 2017

Multiple Different Natural Language Processing Tasks in a Single Deep Model. This, in my opinion, is indeed a very good
paper. It demonstrates how a neural model can be trained from low-level to higher level in a fashion such that lower layers
correspond to word-level tasks and the higher layers correspond to tasks which are performed at sentence level.
The authors also show how to retain the information at lower layers while training the higher layers by successive regularization.
It is also clearly shown that transfer learning is possible where different datasets are exploited simultaneously after jointly
pre-trained for word embeddings. Catastrophic inference is a vrey crucial thing to deal with in this mode.
It is basically the inference in other layer's learned parameters while training a particular layer. As an example, you want to retain information about POS while training for, say, chunking later!

## Model Architecture:

![](images/model.png)

## Data:

* Conll2000 (http://www.cnts.ua.ac.be/conll2000/chunking/)
* SICK data (http://clic.cimec.unitn.it/composes/sick.html)

## Tasks:

* POS Tagging (word-level)
* Chunking (word-level)
* Semantic Relatedness (sentence-level)
* Textual Entailment (sentence-level)

## Usage:

data.py - Preprocesses data for the model

run.py  - Runs the main model.

## Note:

The original paper contains one more task which is dependency parsing. Currently, that is not incorporated in the model due to
non-availability of good public data. Also need to add successive regularization.

## Citations:

A Joint Many-Task Model: Growing a Neural Network for Multiple NLP Tasks
Kazuma Hashimoto, Caiming Xiong, Yoshimasa Tsuruoka, Richard Socher

https://arxiv.org/abs/1611.01587
