#3 - Kategorik Verileri Sayısallaştırılmak

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer



veriler=pd.read_csv("C:/Users/HBA/Desktop/Wissen Akademi Kursu/eksikveriler.csv") #aynı klasorde degilse dosya uzantısını ekle
print(veriler)

# Ülke değişkeninin kategorik verilerinin sayısallaştırılması

ulke=veriler.iloc[:,0:1].values
print(ulke)

le = LabelEncoder()
#ohe = OneHotEncoder(categorical_features="all")
ct = ColumnTransformer([("ulke", OneHotEncoder(), [0])], remainder = 'passthrough')

ulke[:,0] = le.fit_transform(ulke[:,0])
print(ulke)

#ulke=ohe.fit_transform(ulke).toarray()
ulke= ct.fit_transform(ulke) 
print(ulke)


#4 - Verilerin Birleştirilmesi DataFrame yapmak


# İlk Şart dataların indexli olmasıdır. 
# indexli datalarda indexlere göre birleştirme işlemi yapılacaktır.

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer



veriler=pd.read_csv("C:/Users/HBA/Desktop/Wissen Akademi Kursu/eksikveriler.csv") #aynı klasorde değilse dosya uzantısını ekle
print(veriler)

# ulke değişkenin kategorik verilerinin sayısallaştırılması

ulke=veriler.iloc[:,0:1].values
print(ulke)

le = LabelEncoder()
#ohe = OneHotEncoder(categorical_features="all")
ct = ColumnTransformer([("ulke", OneHotEncoder(), [0])], remainder = 'passthrough')

ulke[:,0] = le.fit_transform(ulke[:,0])
print(ulke)

#ulke=ohe.fit_transform(ulke).toarray()
ulke= ct.fit_transform(ulke) 
print(ulke)

print(list(range(22)))

sonuc = pd.DataFrame(data = ulke, index = range(22), columns=['fr','tr','us'])
print(sonuc)

sonuc2 = pd.DataFrame(data = Yas, index = range(22), columns=['boy','kilo','yas'])
print(sonuc2)

cinsiyet=veriler.iloc[:,-1].values
print(cinsiyet)


sonuc3 = pd.DataFrame(data = cinsiyet, index = range(22), columns=['cinsiyet'])
print(sonuc3)




s=pd.concat([sonuc,sonuc2], axis=1)
print(s)

s2=pd.concat([s,sonuc3], axis=1)
print(s2)

