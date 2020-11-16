# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 12:17:44 2020

@author: TUBA_BOZBAYIR
"""

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

veriler=pd.read_csv('Churn_Modelling.csv')

X=veriler.iloc[:,3:13].values
Y=veriler.iloc[:,13].values

print(type(veriler))

#for numeric
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
X[:,1]=le.fit_transform(X[:,1])

le2=LabelEncoder()
X[:,2]=le2.fit_transform(X[:,2])


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ohe = ColumnTransformer([("Geography", OneHotEncoder(), [4])], remainder = 'passthrough')
X=ohe.fit_transform(X).toarray()
X=X[:,1:]

"""
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,random_state=0)

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)

#YSA
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


classifier=Sequential()
classifier.add(Dense(6, kernel_initializer='uniform', activation='relu', input_dim=11))    # giriş katmanı
classifier.add(Dense(6,kernel_initializer='uniform',activation='relu'))                    # ara katmanlar
classifier.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))                 # çıkış katmanı
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


classifier.fit(X_train, y_train, nb_epoch=50)

y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)

print(cm)
"""









