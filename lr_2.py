import autograd.numpy as np
from autograd import grad
from autograd.test_util import check_grads
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

dataset = datasets.load_breast_cancer()
X, y = dataset.data, dataset.target

def sigmoid(x):
    return 1 / (1 + np.exp(-1*x))

def loss(W):
    y_hat = sigmoid(np.dot(X, W))
    prob = y_hat * y + (1 - y_hat) * (1 - y)
    return -1*np.sum(np.log(prob))

gradient = grad(loss)
W = np.array([0.0]*(X.shape[1]))
check_grads(loss, modes=['rev'])(W)
# print(W)
for i in range(1000):
    W = W - gradient(W) * 0.001

def predict(X , W):
    y_predicted = sigmoid(np.dot(X,W))
    y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
    # print(y_predicted_cls)
    # print(y)
    return np.array(y_predicted_cls)
print(accuracy(y,predict(X, W))*100,'%')