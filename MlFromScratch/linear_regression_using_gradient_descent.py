import numpy as np

class LinearRegression:

	def __init__(self, lr = 0.01, ite = 1000):
		self.lr = lr
		self.ite = ite
		self.weights = None
		self.bias = None
 
	def fit(self, X, y):
		training_size, feature_size = X.shape
		self.weights = np.zeros(feature_size)
		self.bias = 0

		for _ in range(self.ite):
			#partial derivative formula for chaning the weights
			# for bias => 1 / n * sum(predicted - expected)
			# for all other => 1 / n * (predicted - expected) * xT
			#since it is predicted - expected we use subtraction to change the weight
			y_predicted = np.dot(X, self.weights.T) + self.bias
			#print("y predicted is : ", y_predicted)

			d1 = 1 / training_size * np.dot(X.T, (y_predicted - y))
			d0 = 1 / training_size * np.sum(y_predicted - y)
			#print(d1)
			self.weights -= self.lr * d1
			self.bias -= self.lr * d0
			#print(self.weights) 

	def predict(self, x):
		return np.dot(x, self.weights.T) + self.bias



# X = [[1], [2]]
# y = [1, 2]
# X = np.asarray(X)
# y = np.asarray(y)

# print(X)
# print(y)

# print(type(X))
#print(type(y))

# linear_regression = LinearRegression(lr = 0.01, ite = 3)
# linear_regression.fit(X, y)


'''
[
	[2, 1, 3, 2]
	[2, 3, 1, 5]
	[3, 4, 8, 6]
]

n * f
f * 1



'''









