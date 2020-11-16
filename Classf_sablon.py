
#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

#2. Veri Onisleme

#2.1. Veri Yukleme
veriler = pd.read_csv('veriler.csv')
#pd.read_csv("veriler.csv")

x = veriler.iloc[:,1:4].values #bağımsız değişkenler
y = veriler.iloc[:,4:].values #bağımlı değişken
print(y)

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

#sınıflandırma algoritmaları
# 1. Logistic Regression
#Bir veya birden fazla bağımsız değişkeni bulunan ve bir sonucu belirlemek için 
#kullanılan istatistik yönetimidir. 
#Var olan bir veri kümesinin analizi sonucu iki olası sonucu bize verir. 
#Doğrusal sınıflandırma problemlerinde kullanılır.
#Lojistik regresyon, ikili(binary) 1 veya 0 olarak kodlanmış verileri içerir.

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train) #egitim

y_pred = logr.predict(X_test) #tahmin
print(y_pred)
print(y_test)

#karmasiklik matrisi
#Gerçekte var olanlarda , sizin tahminlediğiniz verileri bize verir 
cm = confusion_matrix(y_test,y_pred)
print(cm)


# 2. K-NN En yakın Komşu algoritması 
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski') #en yakın bir komşuna bak 
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print(cm)

    
# 3. SVC (Support Vector SVM classifier)
from sklearn.svm import SVC
svc = SVC(kernel='poly')
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('SVC')
print(cm)

# 4. Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('GNB')
print(cm)

# 5. Decision tree
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy')

dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('DTC')
print(cm)

# 6. Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion = 'entropy')
rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('RFC')
print(cm)


# 7. ROC , TPR, FPR değerleri 

y_proba = rfc.predict_proba(X_test) #Bir değerin sınıflandırma olasılıklarını gösterir
print(y_test)
print(y_proba[:,0])

from sklearn import metrics
#true pozitif rate ve False Pozitif rate gösterimi 
fpr , tpr , thold = metrics.roc_curve(y_test,y_proba[:,0],pos_label='e')
print(fpr)
print(tpr)




