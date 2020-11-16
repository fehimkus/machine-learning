import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LinearRegression


veriler = pd.read_csv('satislar.csv')

aylar = veriler[['Aylar']]
print(aylar)
satislar = veriler[['Satislar']]
print(satislar)

x_train, x_test, y_train, y_test = train_test_split(aylar, satislar, test_size=0.33, random_state=0)

lr = LinearRegression()
lr.fit(x_train, y_train)

tahmin = lr.predict(x_test)
print(tahmin)

x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train, y_train)
plt.plot(y_test, lr.predict(x_test))

plt.title('aylara göre satis rakamlari')
plt.xlabel('Aylar')
plt.ylabel('Satislar')

plt.show()

