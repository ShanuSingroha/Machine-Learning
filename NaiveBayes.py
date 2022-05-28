import pandas as  pd
df = pd.read_csv("train.csv")
df.keys()

df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)
df.head()

inputs = df.drop("Survived" , axis = 'columns')
target = df.Survived
inputs.Sex = inputs.Sex.map({ 'male' : 1 , 'female' : 0})
inputs
inputs.Age = inputs.Age.fillna(inputs.Age.mean())
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(inputs,target,test_size=0.3)
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train , y_train)
model.score(X_test , y_test)
