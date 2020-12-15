# LSTM 
LSTMs can be used to generate predictions in text. Below is a simple LSTM using Tensorflow (Keras) to predict sentence endings based off of training data from Dr. Seuss' "The Cat in the Hat", from https://www.oatridge.co.uk/poems/d/dr-seuss-cat-in-the-hat.php. 

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

Compile the model using an optimizer.

```
model.compile(loss='categorical_crossentropy', optimizer='adam')
```

Fit the model based on the generated input and output.

```
model.fit(X, y, epochs=4, batch_size=256, callbacks=desired_callbacks)
```

Change the output from binary to readable characters and use the model to generate a sequence of characters. 

```
num_to_char = dict((i, c) for i, c in enumerate(chars))
tart = numpy.random.randint(0, len(x_data) - 1)
pattern = x_data[start]
```

Predict the characters next in your pattern and print it to terminal.

```
for i in range(1000):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(vocab_len)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = num_to_char[index]
    seq_in = [num_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
```

The genome is the "text of life", and LSTMs lanauge modeling processing can be easily translated from words to the genome. Next we're going to talk about some examples of how LSTMs can be applied to genomic data. 
