
import numpy as np # linear algebra
import pandas as pd # data processing


df = pd.read_csv('multiple-lr-data.csv')

df.head()
y=df['loan']
x=df[['age','credit-rating','children']]
# Encoding categorical data
from sklearn.model_selection import train_test_split

x_trainm, x_testm, y_trainm, y_testm = train_test_split(x,y,test_size=0.33, random_state=0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x,y)

# Predicting the Test set results

y_predm = regressor.predict(x)

