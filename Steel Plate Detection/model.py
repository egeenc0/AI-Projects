import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

input_size = 14
class Model(Sequential):
    def __init__(self):
        super(Model, self).__init__([
            Input(shape=(input_size,)),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            #First change->Added an unit of layer
            Dense(7, activation='sigmoid')
        ])
        self.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

