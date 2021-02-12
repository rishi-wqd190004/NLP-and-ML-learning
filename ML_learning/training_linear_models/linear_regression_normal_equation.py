import numpy as np
import matplotlib.pyplot as plt
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
lr = 0.1
n_iterations = 1000
m = 100
theta = np.random.randn(2,1)
for iternation in range(n_iterations):
    gradients = 2/m * x_b.T.dot(x_b.dot(theta) - y)
    theta = theta - lr * gradients
print('value at GD steps \n', theta)