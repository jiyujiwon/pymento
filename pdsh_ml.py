import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

iris = sns.load_dataset('iris')
# sns.pairplot(iris, hue='species', height=1.5);
X_iris = iris.drop('species', axis=1)
# print(X_iris)
#
y_iris = iris['species']
# print(Y_iris)
# plt.show()
# plt.style.use('seaborn-whitegrid')
#
# rng = np.random.RandomState(42)
# x = 10 * rng.rand(50)
# print(x)
# y = 2 * x - 1 + rng.randn(50)
#
# X = x[:, np.newaxis]
#
# model = LinearRegression(fit_intercept=True)
# model.fit(X, y)
#
#
# xfit = np.linspace(-1, 11)
# Xfit = xfit[:, np.newaxis]
# print(" \n Xfit:", Xfit)
# yfit = model.predict(Xfit)
# plt.scatter(x, y)
# plt.plot(xfit, yfit)
# plt.show()
from sklearn.model_selection import train_test_split

Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris, random_state=1)

from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

y_model = model.predict(Xtest)
print(y_model)
from sklearn.metrics import accuracy_score

print(accuracy_score(ytest, y_model))
