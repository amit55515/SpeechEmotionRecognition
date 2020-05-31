from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from utils import load_data
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

model = KNeighborsClassifier()

# train the model
print("[*] Training the model...")
model.fit(X_train, y_train)

# predict 25% of data to measure how good we are
y_pred = model.predict(X_test)

# calculate the accuracy
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

print("Accuracy: {:.2f}%".format(accuracy*100))

# now we save the model
# make result directory if doesn't exist yet
if not os.path.isdir("result"):
    os.mkdir("result")

pickle.dump(model, open("result/KnearestNeighbour.model", "wb"))