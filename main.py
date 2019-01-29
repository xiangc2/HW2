import numpy as np
import math
from numpy import genfromtxt
from SVM import SVM

CONTINOUS = [2,4,10,11,12]
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
train_size  = math.ceil(data.shape[0]*0.8)
train       = data[:train_size,:]
validation  = data[train_size:,:]
classifier = SVM(dims=5, reg=0.0001, lr=0.001)
classifier.train(data=train, epochs=50, val=validation)
classifier.plot()