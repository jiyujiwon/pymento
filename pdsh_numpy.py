import numpy as np
import pandas as pd
import timeit
from scipy import special
import matplotlib.pyplot as plt
import seaborn;


def Conversion(centi):
    inch = 0.3937 * centi
    feet = 0.0328 * centi
    print("Inches is:", inch)
    print("Feet is:", feet)


# Python program to convert centimeter to feet and
# Inches Function to perform conversion


seaborn.set()  ## For the plot style

data = pd.read_csv('president_heights_new.csv')
x = np.arange(4)
print(x)
print("x+5=", x + 5)
print("x - 5 =", x - 5)
print("x * 2 =", x * 2)
print("x / 2 =", x / 2)
print("x // 2 =", x // 2)  # floor division

# unary ufunc
print("-x     = ", -x)
print("x ** 2 = ", x ** 2)
print("x % 2  = ", x % 2)
print(-(0.5 * x + 1) ** 2)

x = np.array([-2, -1, 0, 1, 2])
print(np.abs(x))

x = np.array([3 - 4j, 4 - 3j, 2 + 0j, 0 + 1j])
print(np.abs(x))

theta = np.linspace(0, np.pi, 3)
print("theta      = ", theta)
print("sin(theta) = ", np.sin(theta))
print("cos(theta) = ", np.cos(theta))
print("tan(theta) = ", np.tan(theta))

x = [1, 2, 3]
print("x     =", x)
print("e^x   =", np.exp(x))
print("2^x   =", np.exp2(x))
print("3^x   =", np.power(3, x))

x = [1, 2, 4, 10]
print("x        =", x)
print("ln(x)    =", np.log(x))
print("log2(x)  =", np.log2(x))
print("log10(x) =", np.log10(x))

print("\nSpecialized Log Functions below\n")
x = [0, 0.001, 0.01, 0.1]
print(x)
print("exp(x) - 1 =", np.expm1(x))
print("log(1 + x) =", np.log1p(x))

print("\n Gamma functions (generalized factorials) and related functions\n ")

x = [1, 5, 10]
print(x)
print("gamma(x)     =", special.gamma(x))
print("ln|gamma(x)| =", special.gammaln(x))
print("beta(x, 2)   =", special.beta(x, 2))

# Error function (integral of Gaussian)
# its complement, and its inverse
print("\n Error Functions \n")
x = np.array([0, .5, .91, .95, .99, 1.0])
print("erf(x)  =", special.erf(x))
print("erfc(x) =", special.erfc(x))
print("erfinv(x) =", special.erfinv(x))

x = np.arange(5)
y = np.empty(5)
np.multiply(x, 10, out=y)
print(y)

y = np.zeros(10)
np.power(2, x, out=y[::2])
print(y)

print("\n Aggregates")
print("--------------------------------------------------------------------------")
x = np.arange(1, 6)
print(x)
print(np.add.accumulate(x))
print(np.add.reduce(x))

print(np.multiply.accumulate(x))
print(np.multiply.reduce(x))

print(np.multiply.outer(x, x))

L = np.random.random(100)
print(sum(L))

big_array = np.random.rand(1000000)

start_time = timeit.default_timer()
print(sum(big_array))
print(timeit.default_timer() - start_time)

start_time = timeit.default_timer()
print(np.sum(big_array))
print(timeit.default_timer() - start_time)

print("Max, Min:", min(big_array), max(big_array))
print("Max, Min:", np.min(big_array), np.max(big_array))

M = np.random.random((3, 4))
print(M)
print(M.sum())
print(np.sum(M, axis=0))
print(np.sum(M, axis=1))

print(data.head())
heights = np.array(data['height(cm)'])
print(heights)

print("\nMean Height")
Conversion(heights.mean())
print("SD Height:", heights.std())

print("\n Max Height:")
Conversion(heights.max())

print("Min Height:", heights.min())

print(" \n 25th percentile:   ", np.percentile(heights, 25))
print("Median:            ", np.median(heights))
print("75th percentile:   ", np.percentile(heights, 75))

# plt.hist(heights * 0.0328)
# plt.title("Height Distribution of US Presidents")
#
# plt.xlabel('height (cm)')
# plt.ylabel('number');

##### Broadcasting examples####
a = np.array([0, 1, 2])
b = np.array([5, 6, 7])
print(a + b)
print(a + 5)

M = np.ones((3, 3))
print(M + a)

a = np.arange(3)
b = np.arange(3)[:, np.newaxis]
print(a)
print(b)
print(a + b)

print("\nExample of broadcasting(Stretching) 1")
a = np.arange(3).reshape(3, 1)
print(a)
b = np.arange(3)
print(b)
print(M + 1)

# Centering an Array #
print("\n Centering array")
X = np.random.random((10, 3))
print(X)
Xmean = X.mean(0)
print(Xmean)
X_Centered = X - Xmean
print(X_Centered)
print(X_Centered.mean(0))

print("\n plotting 2d")
# x and y have 50 steps from 0 to 5
x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 50)[:, np.newaxis]

# z = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)
# plt.imshow(z, origin='lower', extent=[0, 5, 0, 5],
#            cmap='viridis')
# plt.colorbar();

rng = np.random.RandomState(0)
x = rng.randint(10, size=(3, 4))
print(x)
a = object()
b = a
print(a.__eq__(b))

# how many values less than 6 in each row?# how ma
print(np.sum(x < 6, axis=1))
print(np.any(x > 8))

rand = np.random.RandomState(13)
x = rand.randint(100, size=10)
print("\n x=:", x)
ind = [0, 2, 3]
print(x[ind])

mean = [0, 0]
cov = [[1, 2],
       [2, 5]]
X = rand.multivariate_normal(mean, cov, 100)

indices = np.random.choice(X.shape[0], 20, replace=False)
selection = X[indices]
plt.scatter(X[:, 0], X[:, 1])
plt.scatter(selection[:, 0], selection[:, 1], facecolor=None, s=200)

x = np.arange(10)
i = np.array([2, 1, 8, 4])
x[i] = 99
print(x)

np.random.seed(42)
x = np.random.randn(100)
bins = np.linspace(-5, 5, 20)
# plt.gcf().clear()

# plt.hist(x, bins, histtype='step')
# plt.show()


x = np.array([2, 1, 4, 8, 3])
i = np.argsort(x)
print(x[i])

x = rand.randint(0, 10, (4, 6))
print(x, "\n")
print(np.sort(x, axis=1))

a = np.random.randint(10, size=10)
print(a)
p = np.partition(a, 5)
print(p)
plt.gcf().clear()
k = rand.rand(10, 2)
print("\n k values are:",k)

plt.scatter(k[:, 0], k[:, 1], s=100)
print(k[:,np.newaxis,:])
dist_sq = np.sum((k[:, np.newaxis, :] - k[np.newaxis, :, :]) ** 2, axis=-1)
diffs = k[:, np.newaxis, :] - k[np.newaxis, :, :]
nearest = np.argsort(dist_sq, axis=1)
print(nearest)
K = 2
nearest_partition = np.argpartition(dist_sq, K + 1, axis=1)
plt.scatter(k[:, 0], k[:, 1], s=100)

# draw lines from each point to its two nearest neighbors
K = 2

for i in range(k.shape[0]):
    for j in nearest_partition[i, :K + 1]:
        # plot a line from X[i] to X[j]
        # use some zip magic to make it happen:
        plt.plot(*zip(k[j], k[i]), color='black')
plt.show()
