

import numpy as np
import pandas as pd
import re
import nltk

eclothes = pd.read_csv('Womens Clothing E-Commerce Reviews.csv')


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

nltk.download('stopwords')
from nltk.corpus import stopwords

#Preprocessing (Önişleme)
derlem = []
for i in range(1000):
    review = re.sub('[^a-zA-Z]',' ',str(eclothes['Review Text'][i]))
    review = review.lower()
    review = review.split()
    review = [ps.stem(kelime) for kelime in review if not kelime in set(stopwords.words('english'))]
    review = ' '.join(review)
    derlem.append(review)
    
#Feautre Extraction ( Öznitelik Çıkarımı)
#Bag of Words (BOW)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2000)
X = cv.fit_transform(derlem).toarray() # bağımsız değişken
y = eclothes.iloc[:,5:6].values # bağımlı değişken

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)

y_pred = gnb.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)



















