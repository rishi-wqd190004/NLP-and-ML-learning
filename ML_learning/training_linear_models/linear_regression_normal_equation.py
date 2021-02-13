import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

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
theta_path_sgd = []
m = len(x_b)
n_epochs = 50
t0, t1 = 5, 50 # learning schedule parameters
def learning_schedule(t):
    return t0 / (t+t1)

theta = np.random.randn(2,1)

for epoch in range(n_epochs):
    for i in range(m):
        if epoch == 0 and i<20:
            y_pred = x_new_b.dot(theta)
            style = "b-" if i > 0 else "r--"
            plt.plot(x_new, y_pred, style)
        random_index = np.random.randint(m)
        xi = x_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
        theta_path_sgd.append(theta)
print('SGD for 50 iterations: ', theta)

sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1) #eta0 is initial lr; tol is for early stopping
sgd_reg.fit(x, y.ravel())
print('SGD intercept: {} \n SGD coef: {}'.format(sgd_reg.intercept_, sgd_reg.coef_))

## applying Mini-batch GD
theta_path_mgd = []
n_iterations = 50
minibatch_size = 20
np.random.seed(42)
theta = np.random.randn(2,1)
t0, t1 = 200, 1000
## using previous set learning_schedule
t = 0
for epoch in range(n_iterations):
    shuffled_indices = np.random.permutation(m)
    x_b_shuffled = x_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    for i in range(0, m, minibatch_size):
        t += 1
        xi = x_b_shuffled[i:i+minibatch_size]
        yi = y_shuffled[i:i+minibatch_size]
        gradients = 2/minibatch_size * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(t)
        theta = theta - eta * gradients
        theta_path_mgd.append(theta)
print('Mini-batch GD theta value : \n', theta)

# changing to an array
theta_path_bgd = np.array(theta_path_bgd)
theta_path_sgd = np.array(theta_path_sgd)
theta_path_mgd = np.array(theta_path_mgd)

# plotting
plt.figure(figsize=(16,8))
plt.plot(theta_path_sgd[:, 0], theta_path_sgd[:, 1], "r-s", linewidth=1, label="Stochastic")
plt.plot(theta_path_mgd[:, 0], theta_path_mgd[:, 1], "g-+", linewidth=2, label="Mini-batch")
plt.plot(theta_path_bgd[:, 0], theta_path_bgd[:, 1], "b-o", linewidth=3, label="Batch")
plt.legend(loc="upper left", fontsize=16)
plt.xlabel(r"$\theta_0$", fontsize=20)
plt.ylabel(r"$\theta_1$   ", fontsize=20, rotation=0)
plt.axis([2.5, 4.5, 2.3, 3.9])
plt.show()

## Polynomial Regression
np.random.seed(42)
m = 100
x = 6 * np.random.randn(m, 1)
y = 0.5 * x**2 + x + 2 + np.random.randn(m, 1)
plt.figure(figsize=(16, 8))
plt.plot(x, y, "b.")
plt.title("Polynomial Regression")
plt.xlabel('$x_1$')
plt.ylabel('$y$', rotation=0)
plt.axis([-3, 3, 0, 10])
plt.show()
# using sklearn package
poly_features = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly_features.fit_transform(x)
print(x[0])
lin_reg = LinearRegression()
lin_reg.fit(x_poly, y)
print('linear regression intercepts and coeff: \n', lin_reg.intercept_, lin_reg.coef_)
x_new = np.linspace(-3,3,100).reshape(100,1)
x_new_poly = poly_features.transform(x_new)
y_new = lin_reg.predict(x_new_poly)
plt.figure(figsize=(16,8))
plt.plot(x,y,'b.')
plt.plot(x_new, y_new, "r-", linewidth=2, label="Predictions")
plt.title("Prediction on Polynomial")
plt.xlabel('$x_1$')
plt.ylabel("$y$", rotation=0)
plt.legend(loc="upper left", fontsize=14)
plt.axis([-3,3,0,10])
plt.show()

# Polynomial regression with pipeline for 300 degree polynomial
np.random.seed(42)
m = 100
x = 6 * np.random.rand(m, 1) - 3
y = 0.5 * x**2 + x + 2 + np.random.randn(m, 1)

for style, width, degree in (('g-', 1, 300), ('b--', 2, 2), ('r-+', 2, 1)):
    polybig_features = PolynomialFeatures(degree=degree, include_bias=False)
    std_scaler = StandardScaler()
    lin_reg = LinearRegression()
    polynomial_reg = Pipeline([
        ('poly_features', polybig_features),
        ('std_scaler', std_scaler),
        ("lin_reg", lin_reg),
    ])
    polynomial_reg.fit(x, y)
    y_newbig = polynomial_reg.predict(x_new)
    plt.plot(x_new, y_newbig, style, label=str(degree), linewidth=width)
plt.plot(x, y, "b.", linewidth=3)
plt.legend(loc='upper left')
plt.xlabel("$x_1$")
plt.ylabel("$", rotation=0)
plt.title("high degree polynomial plots")
plt.axis([-3, 3, 0, 10])
plt.show()

## Learning curves --> solving the problem of underfitting and overfitting
def plot_learning_curve(model, x, y):
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=10)
    train_errors, val_errors = [], []
    for m in range(1, len(x_train)):
        model.fit(x_train[:m], y_train[:m])
        y_train_pred = model.predict(x_train[:m])
        y_val_pred = model.predict(x_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_pred))
        val_errors.append(mean_squared_error(y_val, y_val_pred))
    plt.plot(np.sqrt(train_errors), "r-+", label="train")
    plt.plot(np.sqrt(val_errors), "b-", label="val")
    plt.legend(loc="upper right")
    plt.xlabel("Training set size")
    plt.ylabel("RMSE")

lin_reg = LinearRegression()
plot_learning_curve(lin_reg, x, y)
plt.axis([0, 80, 0, 3])
plt.show()

# 10th degree polynomial model on the same data
polynomial_reg_10 = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("lin_reg", LinearRegression()),
])
plot_learning_curve(polynomial_reg_10, x, y)
plt.axis([0, 80, 0, 3])
plt.show()