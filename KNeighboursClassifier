from sklearn import  datasets
from sklearn.neighbors import KNeighborsClassifier

# Loading Datasets
iris = datasets.load_iris()
# printing the description
# print(iris.DESCR)
features = iris.data
labels = iris.target
# print(features[0] , labels)

# Training the classifier
clf = KNeighborsClassifier()
clf.fit(features , labels)

preds = clf.predict([[1,1,1,1]])
print(preds)
