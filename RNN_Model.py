from sklearn.metrics import accuracy_score
from utils import load_data
import numpy as np
import os
import pickle

# load RAVDESS dataset
X_train, X_test, y_train, y_test = load_data(test_size=0.25)
# number of samples in training data
print("Number of training samples:", X_train.shape[0])
# number of samples in testing data
print("Number of testing samples:", X_test.shape[0])
# number of features used
# this is a vector of features extracted 
# using utils.extract_features() method
print("Number of features:", X_train.shape[1])
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
# Initialising the RNN
model = Sequential()
# Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50))
model.add(Dropout(0.3))

# Adding the output layer
model.add(Dense(units = 1))

# Compiling the RNN
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
labels = {'sad':0,'happy':1,'angry':2,'neutral':3}
y_train1 = []
y_test1 = []
for i in y_train:
    y_train1.append(labels[i])
for i in y_test:
    y_test1.append(labels[i])
model.fit(x = X_train, y = y_train1, epochs = 400, batch_size = 32)

# train the model
print("[*] Training the model...")
# model.fit(X_train, y_train)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# predict 25% of data to measure how good we are
y_pred = model.predict(X_test)

# calculate the accuracy
p = y_pred.round()
accuracy = accuracy_score(y_true=y_test1, y_pred=p)

print("Accuracy: {:.2f}%".format(accuracy*100))

if not os.path.isdir("result"):
    os.mkdir("result")

pickle.dump(model, open("result/lstm_classifier.model", "wb"))
