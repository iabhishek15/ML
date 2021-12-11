import numpy as np	

class LogisticRegression:

	def __init__(self, lr = 0.01, max_iter = 1000):
		self.weights = None
		self.bias = None
		self.lr = lr
		self.max_iter = max_iter

	def fit(self, X, y):
		training_sample, features = X.shape
		self.bias = 0
		self.weights = np.zeros(features)

		for _ in range(self.max_iter):
			linear_model = np.dot(X, self.weights) + self.bias
			#print(linear_model.shape)
			predicted_y = self.sigmoid(linear_model)
			# print(predicted_y.shape)
			# print(X.shape)
			# print(y.shape)
			# print(training_sample)
			# print(features)
			df1 = 1 / (training_sample) * np.dot(X.T, (predicted_y - y))
			df0 =  1 / (training_sample) * np.sum(predicted_y - y)

			self.weights -= self.lr * df1
			self.bias -= self.lr * df0


	def predict(self, X):
		linear_model = np.dot(X,self.weights) + self.bias
		y_predicted = self.sigmoid(linear_model)
		result  = [1 if y > 0.500000000 else 0 for y in y_predicted]
		return result   

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))




