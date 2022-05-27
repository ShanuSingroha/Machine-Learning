import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
diabetes = datasets.load_diabetes()

# dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])
# print(diabetes.keys())
diabetes_X = diabetes.data[:, np.newaxis, 3 ]


diabetes_X_train = diabetes_X[:-10]
diabetes_X_test = diabetes_X[-30:]


diabetes_y_train = diabetes.target[:-10]
diabetes_y_test = diabetes.target[-30:]

model = linear_model.LinearRegression()
model.fit( diabetes_X_train , diabetes_y_train)

diabetes_y_predicted = model.predict(diabetes_X_test)

print("MEAN SQUARED ERROR: ", mean_squared_error(diabetes_y_test, diabetes_y_predicted))

plt.scatter( diabetes_X_test , diabetes_y_test)
plt.plot( diabetes_X_test , diabetes_y_predicted)

plt.show()
