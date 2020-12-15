# LSTM 
LSTMs can be used to generate predictions in text. Below is a simple LSTM using Tensorflow (Keras) to predict sentence endings based off of training data from Dr. Suess' "The Cat in the Hat", from https://www.oatridge.co.uk/poems/d/dr-seuss-cat-in-the-hat.php. 

Tutorial from Harsh Bansal at https://bansalh944.medium.com/text-generation-using-lstm-b6ced8629b03. 

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

Create a 3-layer LSTM, the first two with 256 units of memory and will store sequences of input data, and the last with 128 units of memory, and will generate a prediction for the next character in the sequence. 

```
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
```

Compile the model using an optimizer ([a link[https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c#:~:text=Adam%20%5B1%5D%20is%20an%20adaptive,for%20training%20deep%20neural%20networks.&text=The%20algorithms%20leverages%20the,learning%20rates%20for%20each%20parameter.] Adam was used). 
```
model.compile(loss='categorical_crossentropy', optimizer='adam')
```
The genome is the "text of life", and LSTMs lanauge modeling processing can be easily translated from words to the genome. Next we're going to talk about some examples of how LSTMs can be applied to genomic data. 
