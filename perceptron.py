import numpy as np
import time
class Perceptron:
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        #self.weights[]
        #print("self_w:",self.w_)
        for _ in range(self.n_iter):
            errors = 0
            #print ("X:",X) 
            #print ("y:",y)
            #print("Iteration:%s"%(_))
            zip_x_y = zip(X, y)
            for xi, target in zip_x_y:
                #guess = 
                update = self.eta * (target - self.predict(xi))
                #print ("iteration:",xi,target,self.predict(xi),update)
                self.w_[1:] += update * xi
                #print("self.w_[1:]:", self.w_[1:])
                self.w_[0] += update
                #print("self.w_[0]:", self.w_[0])
                errors += int(update != 0.0)
                #time.sleep(1)
            self.errors_.append(errors)
            #print("error:",errors)
            #time.sleep(2)
        return self
    def net_input(self, X):
        net_input = np.dot(X, self.w_[1:]) + self.w_[0]
        return net_input
    def predict(self, X):
        #print (X,self.net_input(X))
        return np.where(self.net_input(X) >= 0.0, 1, -1)
