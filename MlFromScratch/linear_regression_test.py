import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

X, y = datasets.make_regression(n_samples = 100, n_features = 1, noise = 4, random_state = 4)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)

#Inspect_data
# fig = plt.figure()
# plt.scatter(X[:, 0], y, color = 'b', marker = "o", s = 30)
# plt.show()


reg = LinearRegression()
reg.fit(X_train, y_train)
reg.predict(X_test)

slope = reg.coef_
intercept = reg.intercept_
#print(slope[0])
#drawing the line from the intercept and the slope



# axes = plt.gca()
# x_vals = []
# for i in range(-3,3):
	# x_vals.append(i)
#x_vals = np.array(axes.get_xlim())
# y_vals = intercept + slope * x_vals
# plt.plot(x_vals, y_vals, '--', color = 'red')



# from linear_regression_using_gradient_descent import LinearRegression
# myreg = LinearRegression(lr = 0.01, ite = 1000)

# myreg.fit(X_train, y_train)
# new_slope = myreg.weights
# new_intercept = myreg.bias
# new_y_vals = new_intercept + new_slope * x_vals
# plt.plot(x_vals, new_y_vals, '--', color = 'green')


#fig = plt.figure()
# plt.scatter(X[:, 0], y, color = 'b', marker = "o", s = 30)
# plt.show()

print(slope)
print(intercept)
# print(new_slope)
# print(new_intercept)


from linear_regression_using_ordinary_learst_square_matrix_formulation import LinearRegression

my_mat_reg = LinearRegression()
my_mat_reg.fit(X, y)
print(my_mat_reg.coef_)
print(my_mat_reg.intercept_)













