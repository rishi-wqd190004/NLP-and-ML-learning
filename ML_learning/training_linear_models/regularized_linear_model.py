import numpy as np
from sklearn.linear_model import Ridge

## Ridge regression
### on closed-form solution
np.random.seed(42)
m = 20
x = 3 * np.random.rand(m, 1)
y = 1 + 0.5 * x + np.random.randn(m, 1) / 1.5
x_new = np.linspace(0,3,100).reshape(100,1)
ridge_reg = Ridge(alpha=1, solver="cholesky", random_state=42)
ridge_reg.fit(x, y)
ridge_reg.predict([[1.5]])
print('Ridge on closed-form solution for 1.5: \n', ridge_reg.predict([[1.5]]))
