from sklearn import datasets
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

iris = datasets.load_iris()
features = iris["data"][:, 3:]
labels = (iris.target == 2).astype(int)

# print(features , labels)

clf = LogisticRegression()
clf.fit(features, labels)

prediction = clf.predict([[2.6]])
print(prediction)

X_new = np.linspace(0,3,1000).reshape(-1,1)
y_prob = clf.predict_proba(X_new)
plt.plot(X_new , y_prob[:,1])
plt.show()

