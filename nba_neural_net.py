
""" !conda update scikit-learn """

#import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPRegressor

import numpy as np
import pandas as pd
import os
import os.path
import shutil


path = "/csvstats/"

all_files = list(os.walk(path))

li = []

for filename in all_files:
    currentdf = pd.read_csv(filename, index_col=None, header=0)
    li.append(currentdf)

df = pd.concat(li, axis=0, ignore_index=True)




# For Pandas's read_csv, use header=0 when you know row 0 is a header row
# df here is a "dataframe":
# df = pd.read_csv('filename.csv', header=0)    # read the file

df.head()                                 # first few lines
df.info()                                 # column details




print("+++ Converting to numpy arrays... +++")
# Data needs to be in numpy arrays - these next two lines convert to numpy arrays
X_data_complete = df.iloc[:,0:64].values        # replace '64' with the index that the points are in 
y_data_complete = df[ '64' ].values        # replace '64' with the index that the points are in 


# this segregates unlabeled data (the games that we want to predict)
# first row in the csv should be the inputs of a game that we want to predict and leave the points blank 

# X_unknown = X_data_complete[0:20,0:64]      
# y_unknown = y_data_complete[0:20]

# X_known = X_data_complete[20:,0:64]
# y_known = y_data_complete[20:]

X_known = X_data_complete[:,0:64] #replace '64' with index with the points
y_known = y_data_complete[:]

#
# we can scramble the remaining data if we want to (we do)
# 
KNOWN_SIZE = len(y_known)
indices = np.random.permutation(KNOWN_SIZE)  # this scrambles the data each time
X_known = X_known[indices]
y_known = y_known[indices] 

#
# from the known data, create training and testing datasets
#
TRAIN_FRACTION = 0.50
TRAIN_SIZE = int(TRAIN_FRACTION*KNOWN_SIZE)
TEST_SIZE = KNOWN_SIZE - TRAIN_SIZE   # not really needed, but...
X_train = X_known[:TRAIN_SIZE]
y_train = y_known[:TRAIN_SIZE]

X_test = X_known[TRAIN_SIZE:]
y_test = y_known[TRAIN_SIZE:]



#
# it's important to keep the input values in the 0-to-1 or -1-to-1 range (WHY?)
#    This is done through the "StandardScaler" in scikit-learn
# 
USE_SCALER = True
if USE_SCALER == True:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)   # Fit only to the training dataframe
    # now, rescale inputs -- both testing and training
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_unknown = scaler.transform(X_unknown)

# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor

mlp = MLPRegressor(hidden_layer_sizes=(5,7,), max_iter=200, alpha=1e-4,
                    solver='sgd', verbose=True, shuffle=True, early_stopping = False, # tol=1e-4, 
                    random_state=None, # reproduceability
                    learning_rate_init=.1, learning_rate = 'adaptive')

print("\n\n++++++++++  TRAINING  +++++++++++++++\n\n")
mlp.fit(X_train, y_train)


print("\n\n++++++++++++  TESTING  +++++++++++++\n\n")
print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))

# let's see the coefficients -- the nnet weights!
# CS = [coef.shape for coef in mlp.coefs_]
# print(CS)

# predictions:
predictions = mlp.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print("\nConfusion matrix:")
print(confusion_matrix(y_test,predictions))

print("\nClassification report")
print(classification_report(y_test,predictions))

# For demonstration of predictions 
#
unknown_predictions = mlp.predict(X_unknown)
print("Unknown predictions:")
print("  Correct values:   CORRECT_SCORE ")
print("  Our predictions: ", unknown_predictions)


if False:
    L = [5.2, 4.1, 1.5, 0.1]
    row = np.array(L)  # makes an array-row
    row = row.reshape(1,4)   # makes an array of array-row
    if USE_SCALER == True:
        row = scaler.transform(row)
    print("\nrow is", row)
    print("mlp.predict_proba(row) == ", mlp.predict_proba(row))

# C = R.reshape(-1,1)  # make a column!
