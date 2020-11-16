# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 16:42:32 2020

@author: TUBA_BOZBAYIR
"""

import numpy as np #as den sonra gelenı kısaltma olarak kullanmak ıcın
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer



veriler = pd.read_csv("eksikveriler.csv") #aynı klasorde degılse dosya uzantısını ekle
print(veriler)


ulke = veriler.iloc[:,0:1].values
print(ulke)

le = LabelEncoder()
#ohe = OneHotEncoder(categorical_features="all")
ohe = ColumnTransformer([("ulke", OneHotEncoder(), [0])], remainder = 'passthrough')



ulke[:,0] = le.fit_transform(ulke[:,0])
print(ulke)

#ulke=ohe.fit_transform(ulke).toarray()
ulke= ohe.fit_transform(ulke) 
print(ulke)
