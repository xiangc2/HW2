import numpy as np
import math
import matplotlib.pyplot as plt

class SVM:
    def __init__(self, dims, reg, lr):
        self.dims   = dims
        self.reg    = reg
        self.lr     = lr
        self.acc    = []
        self.w_magnitude = []

        self.W = np.random.rand(dims)
        self.b = np.random.rand(1)

    def split(self, data):
        X = data[:,0:self.dims]
        y = data[:,self.dims]
        return X, y

    def forward(self, X):
        return X.dot(self.W) + self.b

    def train(self, data, epochs, val):
        num_data = data.shape[0]
        for epoch in range(epochs):
            np.random.shuffle(data)
            hold_out_size = 50
            hold_out = data[:hold_out_size,:]
            train    = data[hold_out_size:,:]
            val_X, val_y = self.split(val)
            X, y         = self.split(train)
            for step in range(300):
                index = np.random.randint(num_data-50)
                x0 = X[index]
                y0 = y[index]
                f = self.forward(x0)
                dW = self.reg*self.W
                db = 0
                if f*y0 < 1:
                    dW += -(y0*x0).T
                    db += -y0
                self.W -= self.lr*dW
                self.b -= self.lr*db
                if step % 30 == 0:
                    predict_y = self.predict(val_X)
                    acc = np.sum(predict_y == val_y)/val_y.shape[0]
                    print("epoch: " + str(epoch) + " step: " + str(step) + " acc: " + str(acc) )
                    self.acc.append(acc)
                    self.w_magnitude.append(np.linalg.norm(self.W))

    def predict(self, X):
        num_data = X.shape[0]
        f = self.forward(X)
        predict_y = np.ones(num_data)
        predict_y[f < 0] = -1.0
        return predict_y

    def plot_acc(self):
        num_acc = len(self.acc)
        x = np.arange(num_acc)*30
        plt.xlabel("steps")
        plt.ylabel("accuracy")
        plt.plot(x, self.acc)

    def plot_w(self):
        num_w = len(self.w_magnitude)
        x = np.arange(num_w)*30
        plt.xlabel("steps")
        plt.ylabel("magnitude of coefficient vector")
        plt.plot(x, self.w_magnitude)


            

        