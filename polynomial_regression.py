import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import operator
data = pd.read_csv('/content/data.csv')
print(data.head())
X = data[['sqft_living']].values
y = data['price'].values
plt.scatter(X, y, color='blue')
plt.title('House Prices vs. Square Feet')
plt.xlabel('Square Feet')
plt.ylabel('Price')
plt.show()
polynomial_features = PolynomialFeatures(degree=2)
X_poly = polynomial_features.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y)
y_poly_pred = model.predict(X_poly)
mse = mean_squared_error(y, y_poly_pred)
r2 = r2_score(y, y_poly_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
plt.scatter(X, y, color='blue')
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(X, y_poly_pred), key=sort_axis)
X_sorted, y_poly_pred_sorted = zip(*sorted_zip)
plt.plot(X_sorted, y_poly_pred_sorted, color='red')
plt.title('Polynomial Regression Fit for House Prices')
plt.xlabel('Square Feet')
plt.ylabel('Price')
plt.show()
X_new = np.array([[1000]])
X_new_poly = polynomial_features.transform(X_new)
y_new_pred = model.predict(X_new_poly)
print('Predicted Prices for new square footage:', y_new_pred)
