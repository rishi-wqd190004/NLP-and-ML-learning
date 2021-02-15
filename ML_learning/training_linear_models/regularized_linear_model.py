import numpy as np
from copy import deepcopy
from sklearn.linear_model import Ridge, SGDRegressor, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

## Ridge regression
### by closed-form solution
np.random.seed(42)
m = 20
x = 3 * np.random.rand(m, 1)
y = 1 + 0.5 * x + np.random.randn(m, 1) / 1.5
x_new = np.linspace(0,3,100).reshape(100,1)
ridge_reg = Ridge(alpha=1, solver="cholesky", random_state=42) # can use solver="sag" which is stochastic average GD
ridge_reg.fit(x, y)
ridge_reg.predict([[1.5]])
print('Ridge by closed-form solution for 1.5: \n', ridge_reg.predict([[1.5]]))

### by gradient descent
sgd_reg = SGDRegressor(penalty="l2", max_iter=1000, tol=1e-3, random_state=42)
sgd_reg.fit(x, y.ravel())
sgd_reg.predict([[1.5]])
print('Ridge by Gradient decent for 1.5: \n', sgd_reg.predict([[1.5]]))

## Lasso regression
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(x, y)
print("Lasso regression predict for 1.5: \n", lasso_reg.predict([[1.5]]))

## Elastic Net
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
elastic_net.fit(x, y)
print("Elastic Net predict for 1.5: \n", elastic_net.predict([[1.5]]))

## Early stopping
np.random.seed(42)
m = 100
x = 6 * np.random.rand(m, 1) - 3
y = 2 + x + 0.5 * x**2 + np.random.randn(m, 1)

x_train, x_val, y_train, y_val = train_test_split(x[:50], y[:50].ravel(), test_size=0.5, random_state=10)

poly_scaler = Pipeline([
    ('poly_features', PolynomialFeatures(degree=90, include_bias=False)),
    ("std_scaler", StandardScaler())
])
x_train_poly_scaled = poly_scaler.fit_transform(x_train)
x_val_poly_scaled = poly_scaler.transform(x_val)

sgd_reg = SGDRegressor(max_iter=1, tol=-np.infty, warm_start= True, penalty=None, learning_rate="constant", eta0=0.0005, random_state=42)
minimum_val_error = float('inf')
best_epoch = None
best_model = None
for epoch in range(1000):
    sgd_reg.fit(x_train_poly_scaled, y_train)
    y_val_pred = sgd_reg.predict(x_val_poly_scaled)
    val_error = mean_squared_error(y_val, y_val_pred)
    if val_error < minimum_val_error:
        minimum_val_error = val_error
        best_epoch = epoch
        best_model = deepcopy(sgd_reg)

## plotting the graph
n_epochs = 5000
train_errors, val_errors = [], []
for epoch in range(n_epochs):
    sgd_reg.fit(x_train_poly_scaled, y_train)
    y_train_pred = sgd_reg.predict(x_train_poly_scaled)
    y_val_pred = sgd_reg.predict(x_val_poly_scaled)
    train_errors.append(mean_squared_error(y_train, y_train_pred))
    val_errors.append(mean_squared_error(y_val, y_val_pred))
best_epoch = np.argmin(val_errors)
best_val_rmse = np.sqrt(val_errors[best_epoch])

plt.annotate('Best Model', xy = (best_epoch, best_val_rmse),
            xytext = (best_epoch, best_val_rmse+1), ha="center",
            arrowprops=dict(facecolor="black", shrink=0.05))
best_val_rmse -= 0.03
plt.plot([0, n_epochs], [best_val_rmse, best_val_rmse], "k:", linewidth=2)
plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Validation set")
plt.plot(np.sqrt(train_errors), "r--", linewidth=2, label="Training set")
plt.legend( fontsize=14)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("RMSE", fontsize=14)
plt.show()