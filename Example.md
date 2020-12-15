# LSTM 
LSTMs can be used to generate predictions in text. Below is a simple LSTM using Tensorflow (Keras) to predict sentence endings based off of training data from Dr. Suess' "The Cat in the Hat", from https://www.oatridge.co.uk/poems/d/dr-seuss-cat-in-the-hat.php. 

Tutorial from Harsh Banal at https://bansalh944.medium.com/text-generation-using-lstm-b6ced8629b03. 

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

Store the number of inputted words, as well as the number of unique vocabulary words from the text.

```
input_len = len(processed_inputs)
vocab_len = len(chars)
```

Define sequence length (number of characters considered) and convert the data to readable form.

```
seq_length = 100
x_data = []
y_data = []

for i in range(0, input_len - seq_length, 1):
    in_seq = words[i:i + seq_length]
    out_seq = words[i + seq_length]
    x.append([char_to_num[char] for char in in_seq])
    y.append(char_to_num[out_seq])
    
n_patterns = len(x_data)
```

Format the input as x, and output as y, and convert the output to binary with one-hot-encoding.

```
X = numpy.reshape(x, (n_patterns, seq_length, 1))
X = X/float(vocab_len)
y = np_utils.to_categorical(y)
```

The genome is the "text of life", and LSTMs lanauge modeling processing can be easily translated from words to the genome. Next we're going to talk about some examples of how LSTMs can be applied to genomic data. 
