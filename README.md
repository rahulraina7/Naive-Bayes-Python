# Naive-Bayes-Python
Bayesian spam or ham Classifer

A very basic implementation of Naive-Bayes classifier for classifying the document as spam or ham.

Dependecies :

PickleDB :

https://pythonhosted.org/pickleDB/

NLTK (for tokenizing)

Dataset used :

https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection

Usage :

python naive.py


Notes:
training for a single dataset should be done only once. If the dataset is already trainer, directly select predict (option : 2) .
Enter the text to be classifed from standard input.

Example :

python naive.py

train or predict : 1 or 2

1

Train complete.. Predict now

>>> Congratulations you won a lottery

Class : spam :: Score : 99.41228501192559%

>> Hi how are you doing today rahul

Class : ham :: Score : 99.96228201578403%
