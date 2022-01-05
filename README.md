# NLP_Intent_recognition_AI
Chatbot for Intent Recognition

![Python 3.9](https://www.python.org/static/community_logos/python-logo-generic.svg)


![Python 2.7](https://img.shields.io/badge/python-2.7-blue.svg)
![Python 3.5](https://img.shields.io/badge/python-3.5-blue.svg)
![License](https://img.shields.io/badge/license-GPLv3-blue.svg)

## Disclaimer

I hold no liability for what you do with this bot or what happens to you by using this bot.

## Dependencies

Install the following packages  using `pip`:

```
import numpy as np
import json
import re
import tensorflow as tf
import random
import spacy
nlp = spacy.load('en_core_web_sm')
```
<br>
Import Json file

```
with open('Intent/Intent.json')as f:
    intents= json.load(f)
```
<br>
<br>

## Preprocessing data

In this part we gonna clean data, split them into inputs and targets tensor, build a tokenizer dictionary and turn sentences into sequences.
The target tensor has a bunch of list with a length of unique title list.

```
def preprocessing(line):
    line = re.sub(r'[^a-zA-z.?!\']', ' ', line)
    line = re.sub(r'[ ]+', ' ', line)
    return line
```
## Tokenizing

This returns a document object that contains tokens. A token is a unit of text in the document, such as individual words and punctuation.
tokenizer create tokens for every word in the data corpus and map them to a index using dictionary.

word_index contains the index for each word

```
# split the input sentences into words
    for token in doc:
        if token.text in tokenizer.word_index:
            sent_seq.append(tokenizer.word_index[token.text])
```
<br>
<br>

## Word embeddings

Word embeddings (also called word vectors) represent each word numerically in such a way that the vector
corresponds to how that word is used or what it means.Vector encodings are learned by considering the context in which the words appear.
Words that appear in similar contexts will have similar vectors. For example, vectors for "leopard", "lion", and "tiger" will
be close together, while they'll be far away from "planet" and "castle".
<br>

```
embed_dim=512
 tf.keras.layers.Embedding(vocab_size, embed_dim),
```

## LSTM Model
<br>
 While developing a DL model, we should keep in mind of key things like Model Architecture, Hyperparmeter Tuning and Performance of the model.
 Reccurent Neural Networks can handle a seqence of data and learn a pattern of input seqence to give either sequence or scalar value as output. 
 In our case, the Neural Network outputs a scalar value prediction.
<br>
<br>

```
# hyperparameters
epochs=50
vocab_size=len(tokenizer.word_index) + 1
embed_dim=512
units=128
target_length=target_tensor.shape[1]
# build RNN Model with tensorflow
```
<br>
 Long Short Term Memory, its a variant of RNN which has memory state cell to learn the context of words which are at further along the text to carry 
 contextual meaning rather than just neighbouring words as in case of RNN.
<br>

```
tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, dropout=0.2)),
```
<br>

## Optimization Algorithm

This notebook uses Adam, optimization algorithm for Gradient Descent.
<br>
```
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
```

