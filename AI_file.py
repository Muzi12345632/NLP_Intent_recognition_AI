import numpy as np
import json
import re
import tensorflow as tf
import random
import spacy
from tensorflow.keras.models import load_model
#from bot_main import wa_bot


nlp = spacy.load('en_core_web_sm')

#print(model.summary())

with open('Intent/Intent.json')as f:
    intents= json.load(f)


def preprocessing(line):
    line = re.sub(r'[^a-zA-z.?!\']', ' ', line)
    line = re.sub(r'[ ]+', ' ', line)
    return line


inputs, targets = [], []
classes = []
intent_doc = {}


for intent in intents['intents']:
    if intent['intent'] not in classes:
        classes.append(intent['intent'])
    if intent['intent'] not in intent_doc:
        intent_doc[intent['intent']] = []

    for text in intent['text']:
        inputs.append(preprocessing(text))
        targets.append(intent['intent'])

    for response in intent['responses']:
        intent_doc[intent['intent']].append(response)



def tokenize_data(input_list):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<unk>')

    tokenizer.fit_on_texts(input_list)

    input_seq = tokenizer.texts_to_sequences(input_list)

    input_seq = tf.keras.preprocessing.sequence.pad_sequences(input_seq, padding='pre')

    return tokenizer, input_seq

# preprocess input data
tokenizer, input_tensor = tokenize_data(inputs)


def create_categorical_target(targets):
    word = {}
    categorical_target = []
    counter = 0
    for trg in targets:
        if trg not in word:
            word[trg] = counter
            counter += 1
        categorical_target.append(word[trg])

    categorical_tensor = tf.keras.utils.to_categorical(categorical_target, num_classes=len(word), dtype='int32')
    return categorical_tensor, dict((v, k) for k, v in word.items())


# preprocess output data
target_tensor, trg_index_word = create_categorical_target(targets)

print('input shape: {} and output shape: {}'.format(input_tensor.shape, target_tensor.shape))
model= load_model('NLP.h5')

def response(sentence):
    sent_seq = []
    doc = nlp(repr(sentence))

    # split the input sentences into words
    for token in doc:
        if token.text in tokenizer.word_index:
            sent_seq.append(tokenizer.word_index[token.text])

        # handle the unknown words error
        else:
            sent_seq.append(tokenizer.word_index['<unk>'])

    sent_seq = tf.expand_dims(sent_seq, 0)
    # predict the category of input sentences
    pred = model(sent_seq)
    pred_class = np.argmax(pred.numpy(), axis=1)

    # choice a random response for predicted sentence
    return random.choice(intent_doc[trg_index_word[pred_class[0]]]), trg_index_word[pred_class[0]]

# chat with bot
print("Note: Enter 'quit' to break the conversation.")
while True:
    input_ = input('You: ')
    if input_.lower() == 'quit':
        break
    res, typ = response(input_)
    print('Bot: {} -- TYPE: {}'.format(res, typ))
    print()
