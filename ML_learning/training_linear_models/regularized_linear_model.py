import numpy as np
from sklearn.linear_model import Ridge, SGDRegressor, Lasso, ElasticNet

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