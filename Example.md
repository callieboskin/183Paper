# LSTM 
LSTMs can be used to generate predictions in text. Below is a simple LSTM using Tensorflow (Keras) to predict sentence endings based off of training data from Dr. Suess' "The Cat in the Hat", from https://www.oatridge.co.uk/poems/d/dr-seuss-cat-in-the-hat.php. Tutorial from https://bansalh944.medium.com/text-generation-using-lstm-b6ced8629b03. 

```
import numpy 
import sys

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

file = open("cat_in_hat.txt", "r")
lines = file.readlines()
```

Next, convert the poem to binary, using one-hot-encoding after splitting the words into seperate objects using the tokenize class. 

```
words=word_tokenize(file)
words=" ".join(words)

chars = sorted(list(set(processed_inputs)))
char_to_num = dict((c, i) for i, c in enumerate(chars))
```

The genome is the "text of life", and LSTMs lanauge modeling processing can be easily translated from words to the genome. Next we're going to talk about some examples of how LSTMs can be applied to genomic data. 
