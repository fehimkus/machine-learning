

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# veri kümesi
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# eğitim ve test kümelerinin bölünmesi
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Ölçekleme
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# SVM
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Tahminler
y_pred = classifier.predict(X_test)

#  Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)


#k-katlamali capraz dogrulama 
from sklearn.model_selection import cross_val_score

basari = cross_val_score(estimator = classifier, X=X_train, y=y_train , cv = 4)
 
# estimator : classifier :Şu an eğitimde SVN kullandım başka bir algoritmada kullanıp onu deneyebilirdim.
# X
# Y
#cv : kaç katlamalı

print(basari.mean())
print(basari.std())



# parametremetre optimizasyonu ve algoritma seçimi
from sklearn.model_selection import GridSearchCV
p = [{'C':[1,2,3,4,5],'kernel':['linear']},#linear 'i içindekilere göre teker teker deneyecek
     {'C':[1,2,3,4,5] ,'kernel':['rbf'],
      'gamma':[1,0.5,0.1,0.01,0.001]} ] #gammayı 0.8 denemiştik şimdi başka değerleri deniyoruz

#GSCV parametreleri
#estimator : sınıflandırma algoritması (neyi optimize etmek istediğimiz)
#param_grid : parametreler/ denenecekler
#scoring: neye göre skorlanacak : örn : accuracy
#cv : kaç katlamalı olacağı
#n_jobs : aynı anda çalışacak iş


gs = GridSearchCV(estimator= classifier, #SVM algoritmasını optimize et
                  param_grid = p,
                  scoring =  'accuracy',
                  cv = 10,
                  n_jobs = -1)

grid_search = gs.fit(X_train,y_train)
eniyisonuc = grid_search.best_score_
eniyiparametreler = grid_search.best_params_

print("en iyi sonuc: ",eniyisonuc)
print(eniyiparametreler)













