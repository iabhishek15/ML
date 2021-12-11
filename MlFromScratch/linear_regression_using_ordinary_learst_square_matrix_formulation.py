import numpy as np

class LinearRegression:

	def __init__(self):
		self.coef_ = None
		self.intercept_ = None

	def fit(self, X_train, y_train):
		#(XT X)-1 XT Y
		Y = y_train.T
		X =  np.insert(X_train, 0, 1, axis = 1)
		# print(Y)
		# print(X)
		#X_inv = np.linalg.inv(X)
		XX_inv = np.linalg.inv(np.dot(X.T, X))
		val =- np.dot(XX_inv, X.T)
		weights = np.dot(val, Y)
		self.coef_ = weights
		self.intercept_ = weights[0]
		self.coef_ = np.delete(self.coef_, 0, axis = None)
		# print(weights)
		# print(self.coef_)
		# print(self.intercept_)

	def predict(self, X, y):
		pass



# X = [[4], [7]]
# Y = [2, 4]
# X = np.asarray(X)
# Y = np.asarray(Y)

# linear_regression = LinearRegression()
# linear_regression.fit(X, Y)







