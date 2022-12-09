import pandas as pd


import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("C:\\Users\\daniel\Desktop\\student-details.csv", sep=";")
#student records containing attributes

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
#obtain selected attributes for regression

predict_score = "G3"


#create diff set of arrays for known values and predicted values
X = np.array(data.drop([predict_score], 1))
y = np.array(data[predict_score])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(acc)

print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
