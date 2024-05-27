import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
df = pd.read_csv('data.csv')
#print(df.head())
#plt.xlabel('sqft_living')
#plt.ylabel('price')
#plt.scatter(df.sqft_living, df.price, color='red', marker='+')
reg = linear_model.LinearRegression()
X = df[['price']]
y = df['sqft_living']
reg.fit(X, y)
predicted_price = reg.predict([[40000]])
print(f"Predicted price for 40000 Price: {predicted_price[0]}")
#plt.plot(df.sqft_living, reg.predict(X), color='blue')
#plt.show()
print(reg.coef_)
print(reg.intercept_)
