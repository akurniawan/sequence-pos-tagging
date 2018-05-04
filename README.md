# sequence-tagging

This repo contains both Maximum Entropy and Bi-LSTM + CRF algorithm for pos tagging task. In order to run the experiment please follow how to install and how to run sections

## Datasets
The data is for Indonesian corpus that can be accessed and downloaded at https://github.com/famrashel/idn-tagged-corpus. Within this repo, the data have been separated into training and testing data, both of which are ready to be used.

## How to Install
All of these experiments was done in python 3.6.
Install the dependencies from requirements.txt

For mac user, follow these steps to install megam as one of the requirements for running the maxent algorithm
1. brew tap brewsci/science
2. brew install megam

For linux user, please follow the instructions on https://github.com/nltk/nltk/wiki/Installing-Third-Party-Software#megam-mega-model-optimization-package

## How to Run
### MaxEnt Algorithm
1. Go to maxent
2. Execute `python maxent.py`

## Deep Learning Algorithm
1. Go to deep-learning
2. Run `python pos_tagger.py`
