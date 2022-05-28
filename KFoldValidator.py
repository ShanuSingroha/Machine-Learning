import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
digits = datasets.load_digits()
X = digits.data
Y = digits.target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)
lr = LogisticRegression(solver='liblinear',multi_class='ovr')
lr.fit(X_train, y_train)
lr.score(X_test, y_test)
svm = SVC()
svm.fit(X_train, y_train)
svm.score(X_test, y_test)
rf = RandomForestClassifier(n_estimators=40)
rf.fit(X_train, y_train)
rf.score(X_test, y_test)

# KFOLD VALIDATION --> cross_val_score

from sklearn.model_selection import KFold
kf = KFold()
kf
for train_index, test_index in kf.split([1,2,3,4,5,6,7,8,9,10]):
    print(train_index, test_index)
def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)
get_score(lr ,  X_train, X_test, y_train, y_test)
get_score(rf,  X_train, X_test, y_train, y_test)
get_score(svm,  X_train, X_test, y_train, y_test)

from sklearn.model_selection import StratifiedKFold
kf = StratifiedKFold(n_splits=3)
kf
for train_index, test_index in kf1.split(X,Y):
    print(train_index, test_index)


from sklearn.model_selection import cross_val_score
cross_val_score(RandomForestClassifier(n_estimators=40),digits.data, digits.target,cv=3)
