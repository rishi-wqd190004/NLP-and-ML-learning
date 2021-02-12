import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor

# random x and y numbers
x = 2 * np.random.rand(100,1)
y = 4 + 3 * x + np.random.randn(100,1) #y=4+3x_1+gaussian's noise
# plotting the above numbers
plt.figure(figsize=(16,8))
plt.scatter(x, y)
plt.xlabel('X')
plt.ylabel('Y', rotation=0)
plt.axis([0, 2, 0, 15])
plt.show()
# computing the normal equation
x_b = np.c_[np.ones((100,1)), x]
theta_best = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)
print('theta best with normal equation: \n', theta_best)
# making predictions
x_new = np.array([[0], [2]])
x_new_b = np.c_[np.ones((2,1)), x_new]
y_pred = x_new_b.dot(theta_best)
print('predicted value: \n', y_pred)
# plotting the above numbers
plt.figure(figsize=(16,8))
plt.plot(x_new, y_pred, "r-")
plt.scatter(x, y)
plt.xlabel('X')
plt.ylabel('Y', rotation=0)
plt.axis([0, 2, 0, 15])
plt.show()

## applying batch GD
np.random.seed(42)
lr = 0.1
n_iterations = 1000
m = 100
theta = np.random.randn(2,1)
for iternation in range(n_iterations):
    gradients = 2/m * x_b.T.dot(x_b.dot(theta) - y)
    theta = theta - lr * gradients
print('value at GD steps \n', theta)
#plotting 
theta_path_bgd = []

def plot_gradient_descent(theta, eta, theta_path=None):
    m = len(x_b)
    plt.plot(x, y, "b.")
    n_iterations = 1000
    for iteration in range(n_iterations):
        if iteration < 10:
            y_predict = x_new_b.dot(theta)
            style = "b-" if iteration > 0 else "r--"
            plt.plot(x_new, y_predict, style)
        gradients = 2/m * x_b.T.dot(x_b.dot(theta) - y)
        theta = theta - eta * gradients
        if theta_path is not None:
            theta_path.append(theta)
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 2, 0, 15])
    plt.title(r"$\eta = {}$".format(eta), fontsize=16)

plt.figure(figsize=(10,4))
plt.subplot(131); plot_gradient_descent(theta, eta=0.02)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.subplot(132); plot_gradient_descent(theta, eta=0.1, theta_path=theta_path_bgd)
plt.subplot(133); plot_gradient_descent(theta, eta=0.5)
plt.show()

## applying SGD
n_epochs = 50
t0, t1 = 5, 50 # learning schedule parameters
def learning_schedule(t):
    return t0 / (t+t1)

theta = np.random.randn(2,1)

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = x_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
print('SGD for 50 iterations: ', theta)

sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1) #eta0 is initial lr; tol is for early stopping
sgd_reg.fit(x, y.ravel())
print('SGD intercept: {} \n SGD coef: {}'.format(sgd_reg.intercept_, sgd_reg.coef_))