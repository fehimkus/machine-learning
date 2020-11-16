# 1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.impute import SimpleImputer

# 2.1. Veri Yukleme
fungi = pd.read_csv('mushrooms.csv')
y = fungi[['class']]
x = fungi[['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing',
           'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
           'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type',
           'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']]

k = x.columns

# 2.2 Veri Onisleme

# ----------------------Missing values filled with most frequent data on this column------------------------------------
# As specified in description missing values filled with the question mark. Also mentioned that it is only in
# "stalk-color" column but, just in case whole dataset the question mark replaced according to most frequent strategy
imputer = SimpleImputer(missing_values='?', strategy='most_frequent', verbose=0)
imputer = imputer.fit(x)
x = imputer.transform(x)
# In this situation any missing data in class column should be replaced with poisonous label as "p". Other strategies
# can cause fatal consequences
imputer2 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='p')
imputer2 = imputer2.fit(y)
y = imputer2.transform(y)

# ----------------------Assigning a numerical value for each categorical value for columns------------------------------
le = LabelEncoder()

for i in range(22):
    a = x[:, i:i + 1]
    a = le.fit_transform(a)
    if i < 1:
        df2 = pd.DataFrame(data=a, index=None, columns=[k[i]])
    else:
        df = pd.DataFrame(data=a, index=None, columns=[k[i]])
        df2 = pd.concat([df2, df], axis=1)

print(df2)

# verilerin egitim ve test icin bolunmesi
x = df2.values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

# verilerin olceklenmesi


sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

# sınıflandırma algoritmaları
# 1. Logistic Regression
# Bir veya birden fazla bağımsız değişkeni bulunan ve bir sonucu belirlemek için
# kullanılan istatistik yönetimidir.
# Var olan bir veri kümesinin analizi sonucu iki olası sonucu bize verir.
# Doğrusal sınıflandırma problemlerinde kullanılır.
# Lojistik regresyon, ikili(binary) 1 veya 0 olarak kodlanmış verileri içerir.


logr = LogisticRegression(random_state=0)
logr.fit(X_train, y_train)  # egitim

y_pred = logr.predict(X_test)  # tahmin
print(y_pred)
print(y_test)

# karmasiklik matrisi
# Gerçekte var olanlarda , sizin tahminlediğiniz verileri bize verir
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 2. K-NN En yakın Komşu algoritması


knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')  # en yakın bir komşuna bak
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)

# 3. SVC (Support Vector SVM classifier)


svc = SVC(kernel='poly')
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print('SVC')
print(cm)

# 4. Naive Bayes


gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print('GNB')
print(cm)

# 5. Decision tree


dtc = DecisionTreeClassifier(criterion='entropy')

dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print('DTC')
print(cm)

# 6. Random Forest


rfc = RandomForestClassifier(n_estimators=10, criterion='entropy')
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print('RFC')
print(cm)

# 7. ROC , TPR, FPR değerleri

y_proba = rfc.predict_proba(X_test)  # Bir değerin sınıflandırma olasılıklarını gösterir
print(y_test)
print(y_proba[:, 0])

# true pozitif rate ve False Pozitif rate gösterimi
fpr, tpr, thold = metrics.roc_curve(y_test, y_proba[:, 0], pos_label='e')
print(fpr)
print(tpr)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
y_pred = lda.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('LDA')
print(cm)
