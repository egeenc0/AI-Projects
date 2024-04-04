import pandas as pd
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np


df = pd.read_csv('heart.csv')


input_dim = (13)


model = Sequential()
model.add(Dense(units=input_dim, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='adam')


model.fit(np.array(df.iloc[:, 0:13]), np.array(df['output']), epochs=10)  # 13 girdi özelliği ve son sütun hedef


def process(prediction):
    if prediction > 0.5:
        return True
    else:
        return False

pred = process(model.predict(np.array(df.iloc[0:1, 0:13])))
print(pred)
