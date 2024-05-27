import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv('/content/data.csv')
print(df)

reg  = linear_model.LinearRegression()
reg.fit(df[['bedrooms','bathrooms','sqft_living']],df[['price','floors']])

reg.coef_
reg.intercept_

reg.predict([[3,2,2800]])