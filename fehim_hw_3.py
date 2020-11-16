""""
----------------INFO----------------------------------------------------------------------------------------------------


Author- Fehim KUŞ

-------------The Dataset------------------------------------------------------------------------------------------------

The following acoustic properties of each voice are measured and included within the CSV:

    meanfreq: mean frequency (in kHz)
    sd: standard deviation of frequency
    median: median frequency (in kHz)
    Q25: first quantile (in kHz)
    Q75: third quantile (in kHz)
    IQR: interquantile range (in kHz)
    skew: skewness (see note in specprop description)
    kurt: kurtosis (see note in specprop description)
    sp.ent: spectral entropy
    sfm: spectral flatness
    mode: mode frequency
    centroid: frequency centroid (see specprop)
    peakf: peak frequency (frequency with highest energy)
    meanfun: average of fundamental frequency measured across acoustic signal
    minfun: minimum fundamental frequency measured across acoustic signal
    maxfun: maximum fundamental frequency measured across acoustic signal
    meandom: average of dominant frequency measured across acoustic signal
    mindom: minimum of dominant frequency measured across acoustic signal
    maxdom: maximum of dominant frequency measured across acoustic signal
    dfrange: range of dominant frequency measured across acoustic signal
    modindx: modulation index. Calculated as the accumulated absolute difference between adjacent measurements
    of fundamental frequencies divided by the frequency range
    label: male or female

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as pl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


voices = pd.read_csv('voice.csv')
voices.head()


sns.countplot(voices['label'], label="Count")
plt.show()


voices.drop('label', axis=1).hist(bins=30, figsize=(9, 9))
pl.suptitle("Her bir feature için dağılım değerleri")
plt.savefig('sounds')
plt.show()


feature_names = ['meanfreq', 'sd', 'median', 'Q25', 'Q75', 'IQR', 'skew', 'kurt', 'sp.ent', 'sfm', 'mode', 'centroid',
                 'meanfun', 'minfun', 'maxfun', 'meandom', 'mindom', 'maxdom', 'dfrange', 'modindx']

X = voices[feature_names]
y = voices['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print('K-NN sınıflandırma eğitim seti: {:.2f}'.format(knn.score(X_train, y_train)))
print('K-NN sınıflandırma test seti: {:.2f}'.format(knn.score(X_test, y_test)))


logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print('Logistic reg eğitim seti : {:.2f}'.format(logreg.score(X_train, y_train)))
print('Logistic reg test seti: {:.2f}'.format(logreg.score(X_test, y_test)))


clf = DecisionTreeClassifier().fit(X_train, y_train)
print('Karar Ağacı eğitim seti: {:.2f}'.format(clf.score(X_train, y_train)))
print('Karar Ağacı test seti: {:.2f}'.format(clf.score(X_test, y_test)))


lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
print(' LDA eğitim seti: {:.2f}'.format(lda.score(X_train, y_train)))
print('LDA test  set: {:.2f}'.format(lda.score(X_test, y_test)))


gnb = GaussianNB()
gnb.fit(X_train, y_train)
print('Bayes eğitim seti: {:.2f}'.format(gnb.score(X_train, y_train)))
print('Bayes test seti: {:.2f}'.format(gnb.score(X_test, y_test)))


svm = SVC()
svm.fit(X_train, y_train)
print(' SVM : {:.2f}'.format(svm.score(X_train, y_train)))
print('SVM: {:.2f}'.format(svm.score(X_test, y_test)))

pred = knn.predict(X_test)
print(confusion_matrix(y_test, pred))
