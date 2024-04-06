import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense,Input,GlobalMaxPooling1D
from tensorflow.keras.layers import LSTM,Embedding
from tensorflow.keras.models import Model

"""
Spam classifier via NLP
"""

df = pd.read_csv('spam.csv',encoding='ISO-8859-1')

df = df.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1)
print(df.head())

df['b_labels'] = df['v1'].map({'ham': 0, 'spam': 1})  # 'labels' yerine 'v1'
Y = df['b_labels'].values

df_train, df_test, Y_train, Y_test = train_test_split(df['v2'], Y, test_size=0.33)  # 'data' yerine 'v2'


MAX_VOCAB_SIZE = 20000
tokenizer = Tokenizer(num_words = MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(df_train)

sequences_train = tokenizer.texts_to_sequences(df_train)
sequences_test = tokenizer.texts_to_sequences(df_test)

word2idx = tokenizer.word_index
V = len(word2idx)
print("Found", word2idx, " tokens")

data_train = pad_sequences(sequences_train)


T = data_train.shape[1]
D = 20
data_test = pad_sequences(sequences_test,maxlen=T)
M = 15

i = Input(shape=(T,))
x = Embedding(V + 1, D)(i)
x = LSTM(M, return_sequences=True)(x)
x = GlobalMaxPooling1D()(x)
x = Dense(1, activation='sigmoid')(x)
model = Model(i,x)

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

r = model.fit(
    data_train, Y_train, epochs=10, validation_data=(data_test, Y_test)
)

arr = np.array(['Hey!Click this to earn free money.'])
data = tokenizer.texts_to_sequences(arr)
data = pad_sequences(data, maxlen=T)
print(model.predict(data))