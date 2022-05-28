import pandas as pd
from sklearn import datasets
digits = datasets.load_digits()
X = digits.data
Y = digits.target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators =  19)
model.fit(X_train , y_train)

model.score(X_train , y_train)
