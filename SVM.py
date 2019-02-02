import numpy as np
import math
import matplotlib.pyplot as plt

class SVM:
    def __init__(self, dims, reg):
        self.dims   = dims
        self.reg    = reg
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

    def train(self, data, seasons):
        num_data = data.shape[0]
        for season in range(seasons):
            np.random.shuffle(data)
            hold_out_size = 50
            hold_out = data[:hold_out_size,:]
            train    = data[hold_out_size:,:]
            ho_X, ho_y = self.split(hold_out)
            X, y       = self.split(train)
            self.lr = 1.0/(0.01*season+50)
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
                    predict_y = self.predict(ho_X)
                    acc = np.sum(predict_y == ho_y)/ho_y.shape[0]
                    self.acc.append(acc)
                    self.w_magnitude.append(np.linalg.norm(self.W))
        acc = np.sum(predict_y == ho_y)/ho_y.shape[0]
        #print("Reg: "+str(self.reg)+" Acc: "+str(acc))

    def predict(self, X):
        num_data = X.shape[0]
        f = self.forward(X)
        predict_y = np.ones(num_data)
        predict_y[f < 0] = -1.0
        return predict_y

    def evaluate(self, data):
        X, y = self.split(data)
        predict_y = self.predict(X)
        acc = np.sum(predict_y == y)/y.shape[0]
        print("reg: " + str(self.reg) + ", acc: " + str(acc))

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
        
    def plot_all(self, reg):
        
        fig = plt.figure(1)
        sub1 = fig.add_subplot(111)
        num_acc = len(self.acc)
        x = np.arange(num_acc)*30
        sub1.set_xlabel("steps")
        sub1.set_ylabel("accuracy")
        sub1.plot(x, self.acc, label=reg)
        sub1.legend()
        fig.savefig('1myimage'+reg+'.jpg', format='jpg', dpi=120)
        #fig.clear()
        
        fig = plt.figure(2)
        sub2 = fig.add_subplot(111)
        num_w = len(self.w_magnitude)
        x = np.arange(num_w)*30
        sub2.set_xlabel("steps")
        sub2.set_ylabel("magnitude of coefficient vector")
        sub2.plot(x, self.w_magnitude, label=reg)
        sub2.legend()
        fig.savefig('2myimage'+reg+'.jpg', format='jpg', dpi=120)
        #fig.clear()


            

        