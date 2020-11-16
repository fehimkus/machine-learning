
#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# veri yukleme
veriler = pd.read_csv('maaslar.csv')

x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]
X = x.values
Y = y.values

#lineer  
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X), color = 'blue')
plt.show()

#polinom  
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
#polinom dönüşümü 2.dereceden 
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly, y)
#aslında oluşturduğun değişkenlere göre üslerini al ve y yi öğren dedim.
plt.scatter(X,Y,color = 'red')
#önce polinom dünyasına çevir sonra çiz
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Pozisyon&Maas Lineer Regresyon')
plt.xlabel('Pozisyon')
plt.ylabel('Maas')

plt.show()

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
#polinom dönüşümü 2.dereceden 
#REGRESYON DERECESİ NE KADAR YÜKSEKSE OKADAR BAŞARILI SONUÇ ALIRIZ
x_poly = poly_reg.fit_transform(X) #öğren komutu
print(x_poly)
print("1")
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Pozisyon&Maas Lineer Regresyon')
plt.xlabel('Pozisyon')
plt.ylabel('Maas')

plt.show()



#tahminler
print("linear tahmin 8 ")
print(lin_reg.predict([[6.5]]))
print("poly_2 tahmin 8")
print(lin_reg2.predict(poly_reg.fit_transform([[6.5]])))

print("linear tahmin 11")
print(lin_reg.predict([[11]]))
print("poly_2 tahmin 11")
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))

