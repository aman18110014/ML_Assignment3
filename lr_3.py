import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from lr_2 import LogisticRegression
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

dataset = datasets.load_breast_cancer()
X, y = dataset.data, dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
model = LogisticRegression(l_r=0.0001, n_iters=1000)
k=3
x_f = np.array_split(X, k)
y_f = np.array_split(y, k)
scores = list()
for i in range(k):
    X_train = list(x_f)
    y_train = list(y_f)
    X_test = X_train.pop(i)
    y_test = y_train.pop(i)
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    scores.append(accuracy(y_test ,predictions))
    print("k =",i+1,": accuracy = ",scores[i]*100,"%")

print("\nL.R accuracy: ", sum(scores)/len(scores)*100,"%")