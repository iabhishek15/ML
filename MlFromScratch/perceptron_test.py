import numpy as np 
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

def accuracy(y_true, y_predicted):
	accuracy_var = np.sum(y_true == y_predicted) / len(y_true)
	return accuracy_var

X, y = datasets.make_blobs(n_samples = 200, centers = 2, n_features = 2, cluster_std = 1.5, random_state = 1234)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 12)



from perceptron import Perceptron 

per = Perceptron(learning_rate = 0.01, iters = 1000)

per.fit(X_train, y_train)
predicted_y = per.predict(X_test)

# print(y_test)
# print(predicted_y)

print('accuracy is : ', accuracy(y_test, predicted_y))





