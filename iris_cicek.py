import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
import sklearn.metrics as sm

iris = datasets.load_iris()
x = pd.DataFrame(iris.data)

x.columns = ['Sepal_length', 'Sepal_width', 'Petal_length', 'Petal_width']
print(x)

y = pd.DataFrame(iris.target)
y.columns = ['Targets']
print(y)

plt.figure(figsize=(14, 7))
colormap = np.array(['red', 'blue', 'black'])

plt.subplot(1, 2, 1)
plt.scatter(x.Sepal_length, x.Sepal_width, c=colormap[y.Targets], s=40)
plt.title('sepal')
plt.show()

plt.subplot(1, 2, 2)
plt.scatter(x.Petal_length, x.Petal_width, c=colormap[y.Targets], s=40)
plt.title('petal')
plt.show()

model = KMeans(n_clusters=3)
model.fit(x)

predY = np.choose(model.labels_, [1, 0, 2]).astype(np.int64)
plt.figure(figsize=(14, 7))
colormap = np.array(['red', 'blue', 'black'])

plt.subplot(1, 2, 2)
plt.scatter(x.Petal_length, x.Petal_width, c=colormap[y.Targets], s=40)
plt.title('petal')
plt.show()

plt.subplot(1, 2, 1)
plt.scatter(x.Sepal_length, x.Sepal_width, c=colormap[predY], s=40)
plt.title('sepal')
plt.show()

print(sm.accuracy_score(y, predY))



