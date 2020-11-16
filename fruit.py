# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 14:17:16 2020

@author: TUBA_BOZBAYIR
"""

import pandas as pd
import matplotlib.pyplot as plt
fruits = pd.read_table('fruit_data_with_colors.txt')
fruits.head()


import seaborn as sns
sns.countplot(fruits['fruit_name'],label="Count")
plt.show()

import pylab as pl
fruits.drop('fruit_label' ,axis=1).hist(bins=30, figsize=(9,9))
pl.suptitle("Her bir feature için dağılım değerleri")
plt.savefig('fruits')
plt.show()


feature_names = ['mass', 'width', 'height', 'color_score']
X = fruits[feature_names]
y = fruits['fruit_label']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print('K-NN sınıflandırma eğitim seti: {:.2f}'
     .format(knn.score(X_train, y_train)))
print('K-NN sınıflandırma test seti: {:.2f}'
     .format(knn.score(X_test, y_test)))

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print('Logistic reg eğitim seti : {:.2f}'
     .format(logreg.score(X_train, y_train)))
print('Logistic reg test seti: {:.2f}'
     .format(logreg.score(X_test, y_test)))


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(X_train, y_train)
print('Karar Ağacı eğitim seti: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Karar Ağacı test seti: {:.2f}'
     .format(clf.score(X_test, y_test)))

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
print(' LDA eğitim seti: {:.2f}'
     .format(lda.score(X_train, y_train)))
print('LDA test  set: {:.2f}'
     .format(lda.score(X_test, y_test)))

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print('Bayes eğitim seti: {:.2f}'
     .format(gnb.score(X_train, y_train)))
print('Bayes test seti: {:.2f}'
     .format(gnb.score(X_test, y_test)))

from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train)
print(' SVM : {:.2f}'
     .format(svm.score(X_train, y_train)))
print('SVM: {:.2f}'
     .format(svm.score(X_test, y_test)))

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
pred = knn.predict(X_test)
print(confusion_matrix(y_test, pred))