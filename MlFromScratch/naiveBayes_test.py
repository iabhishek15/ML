import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd 

X, y = datasets.make_classification(n_samples = 1000, n_features = 10, n_classes = 2, random_state = 1234)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)


x_class1 = np.empty(shape = 0)
x_class2 = np.empty(shape = 0)

for idx, out in enumerate(y_train):
    if int(out) is 0:
        x_class1 = np.append(x_class1, X_train[idx][0])
    else :
        x_class2 = np.append(x_class2, X_train[idx][0])

print(x_class1.shape)
print(x_class2.shape)

one = np.zeros(x_class2.shape[0])
zero = [1 for _ in x_class1]



plt.scatter(x_class1, zero, color = 'orange')
plt.scatter(x_class2, one, color = 'green')
plt.show()


#print(X_train[:, 0].shape)
#print(y_train.shape)
