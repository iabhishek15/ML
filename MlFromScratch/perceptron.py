import numpy as np


class Perceptron:

	def __init__(self, learning_rate = 0.01, iters = 1000):
		self.weights = None
		self.bias = None 
		self.lr = learning_rate
		self.iters = iters
		self.activation_function = self.unit_step_function

	#1*b + w.T * x => predicted_value
	#for updating = lr* predicted_value - actual_value (1 - 0)x
	#using the unit set function if val >= 0 1 else 0

	def fit(self,X, y):
		n_samples, n_features = X.shape
		self.weights = np.zeros(n_features)
		self.bias = 0
		for _ in range(1):
			for idx, x in enumerate(X):
				linear_model = np.dot(x, self.weights) + self.bias
				predicted_y = self.activation_function(linear_model)

				updating_values = self.lr * (predicted_y - y[idx])
				
				self.weights -= updating_values * x
				self.bias -= updating_values
		
	def predict(self,X):
		linear_model = np.dot(X, self.weights) + self.bias
		predicted_value = self.activation_function(linear_model)
		return predicted_value

	def unit_step_function(self, x):
		return np.where(x >= 0, 1, 0)
