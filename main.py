import numpy as np
import math
from numpy import genfromtxt
from SVM import SVM
import os

CONTINOUS = [0,2,4,10,11,12]
TAG = 14 

data = genfromtxt('train.txt',delimiter=',',dtype=str)
X = data[:,CONTINOUS]
y = data[:,TAG]
y[y==" <=50K"] = -1.0
y[y==" >50K"]  = 1.0
X = X.astype(float)
X = X - np.mean(X, axis=0)
X /= np.std(X, axis=0)
data = np.hstack((X,y.reshape(y.shape[0],1))).astype(float)
np.random.shuffle(data)
train_size  = math.ceil(data.shape[0]*0.9)
train       = data[:train_size,:]
validation  = data[train_size:,:]

classifier = SVM(dims=6, reg=0.0001)
classifier.train(data=train, seasons=50)
classifier.plot_all(reg="0.0001")
classifier.evaluate(validation)

classifier = SVM(dims=6, reg=0.001)
classifier.train(data=train, seasons=50)
classifier.plot_all(reg="0.001")
classifier.evaluate(validation)

classifier = SVM(dims=6, reg=0.01)
classifier.train(data=train, seasons=50)
classifier.plot_all(reg="0.1")
classifier.evaluate(validation)

TestData = genfromtxt('test.txt',delimiter=',',dtype=str)
X = TestData[:,CONTINOUS]
X = X.astype(float)
X = X - np.mean(X, axis=0)
X /= np.std(X, axis=0)
predict_y = classifier.predict(X)

file = open("submission.txt", "w")
for i in predict_y:
    if i==-1:
        file.write("<=50K\n")
    else:
        file.write(">50K\n")
file.close()

classifier = SVM(dims=6, reg=0.1)
classifier.train(data=train, seasons=50)
classifier.plot_all(reg="1")
classifier.evaluate(validation)

classifier = SVM(dims=6, reg=1)
classifier.train(data=train, seasons=50)
classifier.plot_all(reg="0.01")
classifier.evaluate(validation)

