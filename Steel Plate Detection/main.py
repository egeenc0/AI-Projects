import pandas as pd

from model import Model

input_size = 21
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

print(train_data.head())
print(train_data.columns)
print(train_data.describe())
print(train_data.dtypes)

X = train_data.iloc[:, :-7]
y = train_data.iloc[:, -7:]
X_test = train_data.iloc[:, :-7]
y_test = train_data.iloc[:, -7:]

# Assuming df is your DataFrame and n is the number of columns you want to drop
X = X.iloc[:, 14:]
X_test = X_test.iloc[:, 14:]
print(X.shape)
print(X_test.shape)

#We took first 28 column as input,last 7 as output.

model = Model()

model.fit(X, y, epochs=20, validation_split=0.2)

test_loss, test_acc = model.evaluate(X_test, y_test)

print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")

"""
Accuracy is pretty low and keeps declining.It must be one of those:

1)Garbage/Wrong data:We took all of the 28 columns as input,even ID's which actually shouldnt affect the
result at all.
2)Model hyperparameters:
a)Size
b)Layer
c)lr
d)Activation function

Now let's troubleshoot.
"""
#Adding one more layer->didnt work
#Lets work with removing first 14 columns of data.
#It increased accuracy significantly.