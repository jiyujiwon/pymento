from datetime import datetime
from dateutil import parser
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from matplotlib import cycler


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

# import pandas as pd
#
# cities = pd.read_csv('data/california_cities.csv')
#
# # Extract the data we're interested in
# lat, lon = cities['latd'], cities['longd']
# population, area = cities['population_total'], cities['area_total_km2']
#
# # Scatter the points, using size and color but no label
# plt.scatter(lon, lat, label=None,
#             c=np.log10(population), cmap='viridis',
#             s=area, linewidth=0, alpha=0.5)
# plt.axis(aspect='equal')
# plt.xlabel('longitude')
# plt.ylabel('latitude')
# plt.colorbar(label='log$_{10}$(population)')
# plt.clim(3, 7)
#
# # Here we create a legend:
# # we'll plot empty lists with the desired size and label
# for area in [100, 300, 500]:
#     plt.scatter([], [], c='k', alpha=0.3, s=area,
#                 label=str(area) + ' km$^2$')
# plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='City Area')
#
# plt.title('California Cities: Area and Population')
# plt.show()


# plt.style.use('classic')
# x = np.linspace(0, 10, 100)
# print(x)
# i = np.sin(x) * np.cos(x[:, np.newaxis])
# print(i)
# plt.imshow(i)
#
# plt.imshow(i,cmap='gray')
# plt.colorbar()
# plt.show()

# from matplotlib.colors import LinearSegmentedColormap
#
#
# def grayscale_cmap(cmap):
#     """Return a grayscale version of the given colormap"""
#     cmap = plt.cm.get_cmap(cmap)
#     colors = cmap(np.arange(cmap.N))
#
#     # convert RGBA to perceived grayscale luminance
#     # cf. http://alienryderflex.com/hsp.html
#     RGB_weight = [0.299, 0.587, 0.114]
#     luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
#     colors[:, :3] = luminance[:, np.newaxis]
#
#     return LinearSegmentedColormap.from_list(cmap.name + "_gray", colors, cmap.N)
#
#
# def view_colormap(cmap):
#     """Plot a colormap with its grayscale equivalent"""
#     cmap = plt.cm.get_cmap(cmap)
#     colors = cmap(np.arange(cmap.N))
#
#     cmap = grayscale_cmap(cmap)
#     grayscale = cmap(np.arange(cmap.N))
#
#     fig, ax = plt.subplots(2, figsize=(6, 2),
#                            subplot_kw=dict(xticks=[], yticks=[]))
#     ax[0].imshow([colors], extent=[0, 10, 0, 1])
#     ax[1].imshow([grayscale], extent=[0, 10, 0, 1])
#
#
# view_colormap('jet')
# plt.show()

# from sklearn.datasets import load_digits
#
# digits = load_digits(n_class=6)
#
# from sklearn.manifold import Isomap
#
# iso = Isomap(n_components=2)
# projection = iso.fit_transform(digits.data)
#
# # plot the results
# plt.scatter(projection[:, 0], projection[:, 1], lw=0.1,
#             c=digits.target, cmap=plt.cm.get_cmap('cubehelix', 6))
# plt.colorbar(ticks=range(6), label='digit value')
# plt.clim(-0.5, 5.5)
#
# plt.show()


# fig = plt.figure()
# ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4], xticklabels=[], ylim=(-1.2, 1.2))
# ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4], xticklabels=[], ylim=(-1.2, 1.2))
# x = np.linspace(0, 10)
# ax1.plot(np.sin(x))
# ax2.plot(np.cos(x))
# plt.show()

# for i in range(1, 7):
#     plt.subplot(2, 3, i)
#     plt.text(0.5, 0.5, str((2, 3, i)), fontsize=11, ha='center')

# fig, ax = plt.subplots(2, 3, sharex='col', sharey='row')
#
# # axes are in a two-dimensional array, indexed by [row, col]
# for i in range(2):
#     for j in range(3):
#         ax[i, j].text(0.5, 0.5, str((i, j)),
#                       fontsize=11, ha='center')


# births = pd.read_csv('data/births.csv')
# quartiles = np.percentile(births['births'], [25, 50, 75])
# mu, sig = quartiles[1], 0.74 * (quartiles[2] - quartiles[0])
# births = births.query('(births>@mu - 5 * @sig) & (births < @mu + 5 * @sig)')
# births['day'] = births['day'].astype(int)
# births.index = pd.to_datetime(10000 * births.year +
#                               100 * births.month +
#                               births.day, format='%Y%m%d')
# births_by_date = births.pivot_table('births',
#                                     [births.index.month, births.index.day])
# births_by_date.index = [pd.datetime(2012, month, day)
#                         for (month, day) in births_by_date.index]
#
# fig, ax = plt.subplots(figsize=(12, 4))
# births_by_date.plot(ax=ax)
#
# # Add labels to the plot
# ax.annotate("New Year's Day", xy=('2012-1-1', 4100), xycoords='data',
#             xytext=(50, -30), textcoords='offset points',
#             arrowprops=dict(arrowstyle="->",
#                             connectionstyle="arc3,rad=-0.2"))
#
# ax.annotate("Independence Day", xy=('2012-7-4', 4250), xycoords='data',
#             bbox=dict(boxstyle="round", fc="none", ec="gray"),
#             xytext=(10, -40), textcoords='offset points', ha='center',
#             arrowprops=dict(arrowstyle="->"))
#
# ax.annotate('Labor Day', xy=('2012-9-4', 4850), xycoords='data', ha='center',
#             xytext=(0, -20), textcoords='offset points')
# ax.annotate('', xy=('2012-9-1', 4850), xytext=('2012-9-7', 4850),
#             xycoords='data', textcoords='data',
#             arrowprops={'arrowstyle': '|-|,widthA=0.2,widthB=0.2', })
#
# ax.annotate('Halloween', xy=('2012-10-31', 4600), xycoords='data',
#             xytext=(-80, -40), textcoords='offset points',
#             arrowprops=dict(arrowstyle="fancy",
#                             fc="0.6", ec="none",
#                             connectionstyle="angle3,angleA=0,angleB=-90"))
#
# ax.annotate('Thanksgiving', xy=('2012-11-25', 4500), xycoords='data',
#             xytext=(-120, -60), textcoords='offset points',
#             bbox=dict(boxstyle="round4,pad=.5", fc="0.9"),
#             arrowprops=dict(arrowstyle="->",
#                             connectionstyle="angle,angleA=0,angleB=80,rad=20"))
#
# ax.annotate('Christmas', xy=('2012-12-25', 3850), xycoords='data',
#             xytext=(-30, 0), textcoords='offset points',
#             size=13, ha='right', va="center",
#             bbox=dict(boxstyle="round", alpha=0.1),
#             arrowprops=dict(arrowstyle="wedge,tail_width=0.5", alpha=0.1));
#
# # Label the axes
# ax.set(title='USA births by day of year (1969-1988)',
#        ylabel='average daily births')
#
# # Format the x axis with centered month labels
# ax.xaxis.set_major_locator(mpl.dates.MonthLocator())
# ax.xaxis.set_minor_locator(mpl.dates.MonthLocator(bymonthday=15))
# ax.xaxis.set_major_formatter(plt.NullFormatter())
# ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter('%h'));
#
# ax.set_ylim(3600, 5400);

#
# fig, ax = plt.subplots(5, 5, figsize=(5, 5))
# fig.subplots_adjust(hspace=0, wspace=0)
#
# # Get some face data from scikit-learn
# from sklearn.datasets import fetch_olivetti_faces
#
# faces = fetch_olivetti_faces().images
#
# for i in range(5):
#     for j in range(5):
#         ax[i, j].xaxis.set_major_locator(plt.NullLocator())
#         ax[i, j].yaxis.set_major_locator(plt.NullLocator())
#         ax[i, j].imshow(faces[int(np.random.randn()) * 10 * i + j], cmap="bone")
#
# plt.show()

# mydefault = plt.rcParams.copy()
# print(mydefault)
#
# # HCA Default Colors in Hex format
# colors = cycler('color', ['#6A737B', '#4097DB', '#D4EFFC', '#D1D3D4', '#C3DBBC', '#488251', '#948272', '#DECBA5'])
# plt.rc('axes', facecolor='#E6E6E6', edgecolor='none', axisbelow=True, grid=True, prop_cycle=colors)
#
# plt.rc('grid', color='w', linestyle='solid')
# plt.rc('xtick', direction='out', color='gray')
# plt.rc('ytick', direction='out', color='gray')
# plt.rc('patch', edgecolor='#E6E6E6')
# plt.rc('lines', linewidth=2)
# x = np.random.randn(1000)
#
# for i in range(6):
#     plt.plot(np.random.rand(10))
# plt.show()
def hist_and_lines():
    np.random.seed(0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(np.random.randn(1000))
    for i in range(3):
        ax[1].plot(np.random.rand(10))
    ax[1].legend(['a', 'b', 'c'], loc='lower left')


# # x = np.linspace(0, 10, 100)
# plt.style.use('presentation')
#
# # print(plt.style.available)
# # plt.plot(x)
# with plt.style.context('fivethirtyeight'):
#     hist_and_lines()
# with plt.style.context('ggplot'):
#     hist_and_lines()
# with plt.style.context('seaborn'):
#     hist_and_lines()
# plt.show()

# from mpl_toolkits import mplot3d
# import mpl_toolkits
# from mpl_toolkits.basemap import Basemap
# import subprocess
# # subprocess.check_call(["python", '-m', 'pip', 'install', 'basemap']) # install pkg
# # subprocess.check_call(["python", '-m', 'pip', 'install',"--upgrade", 'basemap']) # upgrade pkg
#
#
# plt.figure(figsize=(8, 8))
# m = Basemap(projection='ortho', resolution=None, lat_0=50, lon_0=-100)
# m.bluemarble(scale=0.5);
# plt.show()

# # Create some data
# sns.set()
# data = np.random.multivariate_normal([0, 0], [[5, 2], [2, 2]], size=2000)
# data = pd.DataFrame(data, columns=['x', 'y'])
#
# # for col in 'xy':
# #     plt.hist(data[col], normed=True, alpha=0.5)
# for col in 'xy':
#     sns.kdeplot(data[col], shade=True)
# plt.show()

# iris = sns.load_dataset('iris')
# print(iris.head())
#
# sns.pairplot(iris, hue='species', height=2.5)
# plt.show()

# tips = sns.load_dataset('tips')
# print(tips.head())
# tips['tip_pct'] = 100 * tips['tip'] / tips['total_bill']
# # grid = sns.FacetGrid(tips, row='sex', col='time', margin_titles=True)
# # grid.map(plt.hist, "tip_pct", bins=np.linspace(0, 40,15))
# sns.jointplot("total_bill", "tip", data=tips, kind='reg');
# plt.show()
planets = sns.load_dataset('planets')

with sns.axes_style('white'):
    g = sns.catplot("year", data=planets, aspect=4.0, kind='count',
                       hue='method', order=range(2001, 2015))
    g.set_ylabels('Number of Planets Discovered')
plt.show()
