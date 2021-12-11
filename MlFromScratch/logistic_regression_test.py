import numpy as np
import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = datasets.load_breast_cancer()
X,y = df.data, df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 1234)

# l_reg = LogisticRegression(max_iter = 10000)
# l_reg.fit(X_train, y_train)

# print(X_train.shape)
# print(y_train)

def accuracy(pre_y, actual_y):
	return np.sum(pre_y == actual_y) / len(actual_y)

from logistic_regression import LogisticRegression

l_reg = LogisticRegression()
l_reg.fit(X_train, y_train)
predicted_y = l_reg.predict(X_test)
print('accuracy is : ', accuracy(predicted_y, y_test))