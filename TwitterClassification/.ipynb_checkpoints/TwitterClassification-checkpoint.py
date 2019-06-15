import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-notebook')

import re
import string
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Bidirectional, LSTM,  Dropout, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

data = pd.read_csv('Data/train.csv',encoding = "ISO-8859-1")
print(data['Sentiment'].value_counts()/len(data))

punctuation = string.punctuation
def clean(sentence):
    sentence = sentence.lower()
    sentence = sentence.strip('#')
    sentence = re.sub(r'@\w+', '', sentence)
    sentence = re.sub(r'http\S+', '', sentence)
    sentence = ''.join(ch for ch in sentence if ch not in set(punctuation))
    sentence = ''.join([i for i in sentence if not i.isdigit()])
    sentence = ' '.join(sentence.split())
    return sentence
data['SentimentText'] = data['SentimentText'].apply(clean)

WNL = WordNetLemmatizer()
def lemma(sentence):
    s = list()
    for x in sentence.split():
        s.append(WNL.lemmatize(x))
    return ' '.join(s)
data['SentimentText'] = data['SentimentText'].apply(lemma)

X_train, X_val, y_train, y_val = train_test_split(data['SentimentText'], data['Sentiment'], test_size=0.1, random_state=37)
print('# Train data samples:', X_train.shape[0])
print('# Validation data samples:', X_val.shape[0])
assert X_train.shape[0] == y_train.shape[0]
assert X_val.shape[0] == y_val.shape[0]

NB_WORDS = 10000  # Parameter indicating the number of words we'll put in the dictionary
tk = Tokenizer(num_words=NB_WORDS,
               filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
               lower=True,
               split=" ")
tk.fit_on_texts(X_train)

X_train_seq = tk.texts_to_sequences(X_train)
X_val_seq = tk.texts_to_sequences(X_val)

seq_lengths = data['SentimentText'].apply(lambda x: len(x.split(' ')))
MAX_LEN = seq_lengths.max()
X_train_seq_trunc = pad_sequences(X_train_seq, maxlen=MAX_LEN)
X_val_seq_trunc = pad_sequences(X_val_seq, maxlen=MAX_LEN)

model = Sequential()
model.add(Embedding(NB_WORDS, 8, input_length=MAX_LEN))
model.add(Bidirectional(LSTM(16)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

print('Shape of the data Train, Validation on X,y')
print((X_train_seq_trunc.shape, y_train.shape), (X_val_seq_trunc.shape,y_val.shape))

history = model.fit(X_train_seq_trunc, y_train,
                   epochs=10,
                   batch_size=512,
                   validation_data=(X_val_seq_trunc,y_val))

def eval_metric(history, metric_name):
    metric = history.history[metric_name]
    val_metric = history.history['val_' + metric_name]

    e = range(1, 10 + 1)

    plt.plot(e, metric, 'bo', label='Train ' + metric_name)
    plt.plot(e, val_metric, 'b', label='Validation ' + metric_name)
    plt.legend()
    plt.show()

eval_metric(history, 'acc')
eval_metric(history, 'loss')

''' It seems that our builded model is not good enough but no the problem is with embeddings so to improve i suggest folow the following steps:-
 1. Download the glove model from (https://nlp.stanford.edu/projects/glove/) i.e glove.twitter.27B.zip
 2. Extract the txt file with 100 dimensions (It will be enough and efficient more than 1000 dimension of ours) glove.twitter.27B.100d.txt
 3. Place txt file with your code and uncomment the following lines of code
''' 
'''
glove_file = 'glove.twitter.27B.' + str(GLOVE_DIM) + 'd.txt'
emb_dict = {}
glove = open(input_path / glove_file)
for line in glove:
    values = line.split()
    word = values[0]
    vector = np.asarray(values[1:], dtype='float32')
    emb_dict[word] = vector
glove.close()

'''
#To feed this into an Embedding layer, we need to build a matrix containing the words in the tweets and their representative word embedding.
#So this matrix will be of shape (NB_WORDS=1000, GLOVE_DIM=100)
'''
emb_matrix = np.zeros((NB_WORDS, GLOVE_DIM))

for w, i in tk.word_index.items(): #tk is tokenized from previous model
    # The word_index contains a token for all words of the training data so we need to limit that
    if i < NB_WORDS:
        vect = emb_dict.get(w)
        # Check if the word from the training data occurs in the GloVe word embeddings
        # Otherwise the vector is kept with only zeros
        if vect is not None:
            emb_matrix[i] = vect
    else:
        break
'''
# All the parameters contain same meaning from previous apply on same model but with new GLOVE Dimension
'''
model = Sequential()
model.add(Embedding(NB_WORDS, GLOVE_DIM, input_length=MAX_LEN))
model.add(Bidirectional(LSTM(16)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.layers[0].set_weights([emb_matrix])
model.layers[0].trainable = False
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train_seq_trunc, y_train,
                   epochs=10,
                   batch_size=512,
                   validation_data=(X_val_seq_trunc,y_val))
'''
# You would definetly get a more accurate version of same model with Glove (word vector representation)

