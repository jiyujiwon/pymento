import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

titanic = sns.load_dataset('titanic')
from pdsh_pandas import display

planets = sns.load_dataset('planets')
print(planets.shape)
print(planets.head)

rng = np.random.RandomState(42)
ser = pd.Series(rng.rand(10))
print(ser.sum())
print(ser.mean())

df = pd.DataFrame({'A': rng.rand(5), 'B': rng.rand((5))})
print(df.mean(axis=1))
print(planets.dropna().describe())

df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A'],
                   'data1': rng.rand(10)}, columns=['key', 'data1'])

print(df)
print(df.groupby('key').sum())
grpobj = planets.groupby('method')
print(grpobj['orbital_period'].mean())

for (method, group) in planets.groupby('method'):
    print("{0:30s} shape={1}".format(method, group.shape))
print(planets.groupby('method')['year'].describe())

data2 = pd.DataFrame(rng.rand(10))
data2.columns = ['data2']
print(data2)

df = pd.merge(df, data2, left_index=True, right_index=True)
print(df)
print(df.groupby('key').aggregate(['min', np.median, max]))

df.groupby('key').aggregate({'data1': 'min',
                             'data2': 'max'})


def filter_func(x):
    return x['data2'].std() > 0.5


print("\n lambda function \n")

df.groupby('key').transform(lambda x: x - x.mean())
print(df)

decade = 10 * (planets['year'] // 10)
decade = decade.astype(str) + 's'
decade.name = 'decade'
print(planets.groupby(['method', decade])['number'].sum().unstack().fillna(0))
print(titanic.head())
print(titanic.groupby('sex')[['survived']].mean())

print("\n -------------Titanic------------------------------------------  \n")
print(titanic.groupby(['sex', 'class'])['survived'].aggregate('mean').unstack())

print("\n -------------Titanic Pivot------------------------------------  \n")
print(titanic.pivot_table('survived', index='sex', columns='class'))

age = pd.cut(titanic['age'], [0, 18, 80])
print("\n -------------Titanic Pivot and age------------------------------------  \n")
st = titanic.pivot_table('survived', ['sex', age], 'class')
print(st)

print("\n -------------Titanic Pivot and age with fair------------------------------------  \n")
qfair = pd.qcut(titanic['fare'], 2)
qdd = titanic.pivot_table('survived', ['sex', age], [qfair, 'class'])
print(qdd)

print("\n -------------Titanic Pivot and agg func------------------------------------  \n")
af = titanic.pivot_table(index='sex', columns='class', aggfunc={'survived': sum, 'fare': 'mean'})
print(af)
print('\n')
at = titanic.pivot_table('survived', index='sex', columns='class', margins=True)

print(at)

print('\n -------------------------births data-------------------------------\n')
births = pd.read_csv("births.csv")
print(births)
births['decade'] = 10 * (births['year'] // 10)
bpt = births.pivot_table('births', index='decade', columns='gender', aggfunc='sum')
print(bpt)

sns.set()  # uses sns styles
births.pivot_table('births', index='year', columns='gender', aggfunc='sum').plot()
plt.ylabel('Total Births per year');

print('\n -------------------------Sigma clipping-------------------------------\n')
quartiles = np.percentile(births['births'], [25, 50, 75])
mu = quartiles[1]
sig = .74 * (quartiles[2] - quartiles[0])
births = births.query('(births > @mu - 5 * @sig) & (births < @mu + 5 * @sig)')

# set 'day' column to integer; it originally was a string due to nulls
births['day'] = births['day'].astype(int)
# create a datetime index from the year, month, day
births.index = pd.to_datetime(10000 * births.year +
                              100 * births.month +
                              births.day, format='%Y%m%d')

births['dayofweek'] = births.index.dayofweek

births.pivot_table('births', index='dayofweek',
                   columns='decade', aggfunc='mean').plot()
plt.gca().set_xticklabels(['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun'])
plt.ylabel('mean births by day');

births_by_date = births.pivot_table('births',
                                    [births.index.month, births.index.day])

births_by_date.index = [pd.datetime(1980, month, day)
                        for (month, day) in births_by_date.index]

births_by_date.index.name = 'day'
print(births_by_date)

print(births_by_date[births_by_date.index == '1980-03-27'])
