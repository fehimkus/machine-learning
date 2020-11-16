"""
----------------INFO----------------------------------------------------------------------------------------------------
These codes include controlling predictions via the r ^ 2 method for the labeled species whether they are 
edible or poisonous using various properties of mushrooms by linear regression, polynomial approximation 
decision tree and random forest methods.

Author- Fehim KUŞ
----------------Mushrooms features abbreviations------------------------------------------------------------------------
Attribute Information: (classes: edible=e, poisonous=p)
cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s
cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y
bruises: bruises=t,no=f
odor: almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s
gill-attachment: attached=a,descending=d,free=f,notched=n
gill-spacing: close=c,crowded=w,distant=d
gill-size: broad=b,narrow=n
gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y
stalk-shape: enlarging=e,tapering=t
stalk-root: bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,---------missing=?----------
stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
veil-type: partial=p,universal=u
veil-color: brown=n,orange=o,white=w,yellow=y
ring-number: none=n,one=o,two=t
ring-type: cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z
spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y
population: abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y
habitat: grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d
"""
# ----------------------Import necessary Libraries----------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# ----------------------Reading command for csv file and attach with variables------------------------------------------
fungi = pd.read_csv('mushrooms.csv')
x = fungi[['class']]
y = fungi[['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing',
           'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
           'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type',
           'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']]

k = y.columns

# ----------------------Missing values filled with most frequent data on this column------------------------------------
# As specified in description missing values filled with the question mark. Also mentioned that it is only in
# "stalk-color" column but, just in case whole dataset the question mark replaced according to most frequent strategy
imputer = SimpleImputer(missing_values='?', strategy='most_frequent', verbose=0)
imputer = imputer.fit(y)
y = imputer.transform(y)
# In this situation any missing data in class column should be replaced with poisonous label as "p". Other strategies
# can cause fatal consequences
imputer2 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='p')
imputer2 = imputer2.fit(x)
x = imputer2.transform(x)
print(type(x))
# ----------------------Assigning a numerical value for each categorical value for columns------------------------------
le = LabelEncoder()
# ct = ColumnTransformer([("class", OneHotEncoder(), [0])], remainder='passthrough')

x = le.fit_transform(x)
df2 = pd.DataFrame(data=x, index=None, columns=['class'])

for i in range(22):
    a = y[:, i:i + 1]
    a = le.fit_transform(a)
    # a = ct1.fit_transform(a)
    df = pd.DataFrame(data=a, index=None, columns=[k[i]])
    df2 = pd.concat([df, df2], axis=1)


# ----------------------Splitting data to test and train----------------------------------------------------------------
p = df2[['class']]
f = df2[['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing',
         'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
         'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type',
         'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']]

poisonous = p.values  # ---------numpy.ndarray
features = f.values  # ---------numpy.ndarray

p_train, p_test, f_train, f_test = train_test_split(p, f, test_size=0.3, random_state=0)

# ----------------------linear and polynomial regression with r2 control method-----------------------------------------
lr1 = LinearRegression()

lr1.fit(f_train, p_train)
f_pred2 = lr1.predict(f_test)

print(r2_score(p_test, f_pred2))

model = PolynomialFeatures(degree=2)          # not applicable for more than second degree, why?
lr2 = LinearRegression()
f_ = model.fit_transform(f)
f_test_ = model.fit_transform(f_test)
lr2.fit(f_, p)
f_pred1 = lr2.predict(f_test_)

print(r2_score(p_test, f_pred1))

# ----------------------Decision tree-----------------------------------------------------------------------------------
std_scl = StandardScaler()
p_scaled = std_scl.fit_transform(poisonous)
f_scaled = std_scl.fit_transform(features)

svr_reg = SVR(kernel='rbf')
svr_reg.fit(f_scaled, p_scaled)

print(r2_score(p_scaled, svr_reg.predict(f_scaled)))

dtr = DecisionTreeRegressor(random_state=0)
dtr.fit(features, poisonous)

# ----------------------Random Forest regressor-------------------------------------------------------------------------
rfr = RandomForestRegressor(n_estimators=10, random_state=0)
rfr.fit(features, poisonous)

print(r2_score(poisonous, rfr.predict(features)))
# ----------------------------------------------------------------------------------------------------------------------
