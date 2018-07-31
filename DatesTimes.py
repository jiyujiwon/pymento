from datetime import datetime
from dateutil import parser
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm

# plt.style.use('classic')
# x = np.linspace(0, 10, 100)
#
# fig, ax = plt.subplots(2)
#
# ax[0].plot(x, np.sin(x))
# ax[1].plot(x, np.cos(x))
# plt.show()

# plt.style.use('seaborn-whitegrid')
# fig = plt.figure()
# ax = plt.axes()
# x = np.linspace(0, 10, 30)
# y = np.sin(x)

# print(x)
# # ax.plot(x, np.sin(x))
# plt.plot(x, np.sin(x))
#
# plt.plot(x, np.sin(x - 0), color='blue', label='sin(x-0)')
# plt.plot(x, np.sin(x - 1), color='g', label='sin(x-1)')
# plt.plot(x, np.sin(x - 2), color='.75')
# plt.plot(x, np.sin(x - 3), color='#FFDD44')
# plt.plot(x, x + 0, linestyle='solid', label='x+0')
# plt.plot(x, x + 1, linestyle='dashed')
# plt.plot(x, x + 2, '--c')
# plt.xlim(-1, 11)
# plt.ylim(-1.5, 1.5)
#
# plt.title("Random sine waves")
# plt.xlabel("x value")
# plt.ylabel("sin(x)")
# plt.legend()
#
# plt.show()

# plt.plot(x, y, 'o', color='black')

# rng = np.random.RandomState(0)
# for marker in ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']:
#     plt.plot(rng.rand(5), rng.rand(5), marker,
#              label="marker='{0}'".format(marker))
# plt.legend(numpoints=1)
# plt.xlim(0, 1.8)
# plt.plot(x, y, '-ok')

# rng = np.random.RandomState(0)
# x = rng.randn(100)
# print(x)
# y = rng.randn(100)
# colors = rng.rand(100)
# sizes = 1000 * rng.rand(100)
# print(sizes)
#
# plt.scatter(x, y, c=colors, s=sizes, alpha=0.3,
#             cmap='viridis')
# plt.colorbar();  # show color scale
# plt.show()
#
# from sklearn.datasets import load_iris
# iris = load_iris()
# features = iris.data.T
# print(features)
# plt.scatter(features[0], features[1], alpha=0.2,
#             s=100*features[3], c=iris.target, cmap='viridis')
# plt.xlabel(iris.feature_names[0])
# plt.ylabel(iris.feature_names[1])
# plt.show()

# x = np.linspace(0, 10, 50)
# dy = 0.5
# y = np.sin(x) + dy * np.random.randn(50)
#
# plt.errorbar(x, y, yerr=dy, fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0);
# plt.show()
#
# from sklearn.gaussian_process import GaussianProcess
#
# # define the model and draw some data
# model = lambda x: x * np.sin(x)
# xdata = np.array([1, 3, 5, 6, 8])
# ydata = model(xdata)
# print(ydata)
#
# # Compute the Gaussian process fit
# gp = GaussianProcess(corr='cubic', theta0=1e-2, thetaL=1e-4, thetaU=1E-1,
#                      random_start=100)
# gp.fit(xdata[:, np.newaxis], ydata)
#
# xfit = np.linspace(0, 10, 1000)
#
# yfit, MSE = gp.predict(xfit[:, np.newaxis], eval_MSE=True)
#
# dyfit = 2 * np.sqrt(MSE)  # 2*sigma ~ 95% confidence region
# plt.plot(xdata, ydata, 'or')
# plt.plot(xfit, yfit, '-', color='gray')
#
# plt.fill_between(xfit, yfit - dyfit, yfit + dyfit,
#                  color='gray', alpha=0.2)
# plt.xlim(0, 10);
#
# plt.show()
#
# def f(x, y):
#     return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)
#
#
# x = np.linspace(0, 5, 50)
# y = np.linspace(0, 5, 40)
#
# X,Y=np.meshgrid(x,y)
# Z=f(X,Y)
#
# plt.contour(X,Y,Z,20,cmap='RdGy')
# plt.show()


# plt.style.use('seaborn-white')
# data = np.random.randn(1000)
# print(data)
# plt.hist(data, bins=10, normed=True, alpha=.5, histtype='stepfilled', color='steelblue', edgecolor='none')
# plt.show()


# x1 = np.random.normal(16, 18, 1000)
# x2 = np.random.normal(8, 16, 1000)
# x3 = np.random.normal(4, 8, 1000)
# kwargs = dict(histtype='stepfilled', alpha=.3, bins=40)
# plt.hist(x1, **kwargs)
# plt.hist(x2, **kwargs)
# plt.hist(x3, **kwargs)
#
# counts, bin_edges = np.histogram(x1, bins=4)
# print(counts)
# plt.show()


# mean = [0, 0]
# cov = [[1, 1], [1, 2]]
# x, y = np.random.multivariate_normal(mean, cov, 10000).T
# plt.hist2d(x, y, bins=20, cmap='Blues')
# cb = plt.colorbar()
# cb.set_label('counts in bin')
# plt.show()

#
# plt.style.use('classic')
# x = np.linspace(0, 10, 100)
# fig, ax = plt.subplots()
# ax.plot(x, np.sin(x), '-b', label='Sin')
# ax.plot(x, np.cos(x), '--r', label='Cos')
# ax.axis('equal')
# leg = ax.legend()
# ax.legend(loc='upper left', frameon=False)
# ax.legend(frameon=False, loc='lower center', ncol=2)
# ax.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)


plt.show()
