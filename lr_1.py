  
import numpy as np
class LogisticRegression:

    def __init__(self, l_r=0.001, n_iters=2000):
        self.lr = l_r
        self.n_iters = n_iters
        self.W = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.b = 0
        self.W = np.zeros(n_features)
        
        # gradient descent
        for _ in range(self.n_iters):
            y_hat = self.sigmoid(np.dot(X, self.W) + self.b)
            
            dw = (1 / n_samples) * np.dot(X.T, (y_hat - y))  # compute gradients
            db = (1 / n_samples) * np.sum(y_hat - y)
            self.W = self.W - self.lr * dw
            self.b = self.b - self.lr * db

    def predict(self, X):
        y_hat = self.sigmoid(np.dot(X, self.W) + self.b)
        y_hat_cls = [1 if i > 0.5 else 0 for i in y_hat]
        return np.array(y_hat_cls)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))