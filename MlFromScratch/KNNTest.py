import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#FF0000','#00FF00', '#0000FF'])


#now using iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)


#This is the training feature
#print(X)
#This is the output
#print(y)

#print(X_train[0])
#print(X_test[0])
#print(y_train[0])
#print(y_test[0])

#print(X_train.shape)
#print(type(X_train))



#plt.figure()
#plt.scatter(X[:,0], X[:, 1], c = y, cmap = cmap, edgecolor = 'k', s = 20)
#plt.show()


def accuracy(y_pre, y_true):
    accuracy = np.sum(y_true == y_pre) / len(y_true)
    return accuracy



k = 3
from knn import KNN
clf = KNN(K = k)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
#print(predictions)
print("custom KNN classification accuracy", accuracy(predictions, y_test))

















