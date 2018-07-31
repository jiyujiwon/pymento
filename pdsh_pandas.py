import pandas as pd
import numpy as np
import os
import subprocess

print(pd.__version__)
print(np.__version__)

cls = lambda: print('\n' * 10)

data = pd.Series([.25, .5, .75, 1])
print(data)

data = pd.Series([0.25, 0.5, 0.75, 1.0],
                 index=['a', 'b', 'c', 'd'])
print(data['b'])

population_dict = {'California': 38332521,
                   'Texas': 26448193,
                   'New York': 19651127,
                   'Florida': 19552860,
                   'Illinois': 12882135}

population = pd.Series(population_dict)
print(population)

s1 = pd.Series({2: 'a', 1: 'b', 3: 'c'})
print(s1)

area_dict = {'California': 423967, 'Texas': 695662, 'New York': 141297,
             'Florida': 170312, 'Illinois': 149995}
area = pd.Series(area_dict)
print(area)
states = pd.DataFrame({'population': population, 'area': area})

print(states)

x = pd.DataFrame(np.random.rand(3, 2),
                 columns=['foo', 'bar'],
                 index=['a', 'b', 'c'])

print(x)

ind = pd.Index([2, 3, 5, 6, 10])

print(ind[:3])
print(ind.size, ind.shape, ind.ndim, ind.dtype)

indA = pd.Index([11, 33, 55, 77, 99])
indB = pd.Index([22, 33, 55, 77, 1111])
print(indA & indB)
print(indA | indB)
print(indA ^ indB)

print(" \n A new Chapter of Slicing and dicing Python Pandas DS \n")
data = pd.Series([0.25, 0.5, 0.75, 1.0],
                 index=['a', 'b', 'c', 'd'])
print('a' in data)
print(data.keys())
print(list(data.items()))
data['e'] = 1.25

print(data.keys())
print(list(data.items()))
print(" \n slicing using index \n")
print(data[0:2])
print(data['a':'b'])
print(data[(data > 0.3) & (data < 1)])

data = pd.Series(['a', 'b', 'c'], index=[1, 3, 5])
print(data)
print(data[1])
print(data[1:3])

### iloc is implicit index, loc uses explicit indexing ###
print("\n Loc attribute \n")
print(data.loc[1:3])
print(data.iloc[1:3])
data = pd.DataFrame({'area': area, 'pop': population})
print(data)
print(data.area)
cls()

data['density'] = data['pop'] / data['area']
print(data)
print(data.values[0])
cls()
rng = np.random.RandomState(42)
ser = pd.Series(rng.randint(0, 10, 4))
print(ser)

df = pd.DataFrame(rng.randint(0, 10, (3, 4)), columns=['A', 'B', 'C', 'D'])
print(df)

print(np.sin(df * np.pi / 4))
cls()

vals1 = np.array([1, None, 3, 4])
print(vals1)

df = pd.DataFrame([[1, np.nan, 2],
                   [2, 3, 5],
                   [np.nan, 4, 6]])

df[3] = np.nan
print(df)

print(df.dropna(axis='rows', thresh=3))


def make_df(cols, ind):
    """Quickly make a DataFrame"""
    data = {c: [str(c) + str(i) for i in ind]
            for c in cols}
    return pd.DataFrame(data, ind)


# example DataFrame
make_df('ABC', range(3))


class display(object):
    """Display HTML representation of multiple objects"""
    template = """<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>{0}</p>{1}
    </div>"""

    def __init__(self, *args):
        self.args = args

    def _repr_html_(self):
        return '\n'.join(self.template.format(a, eval(a)._repr_html_())
                         for a in self.args)

    def __repr__(self):
        return '\n\n'.join(a + '\n' + repr(eval(a))
                           for a in self.args)


cls()
display()

df1 = make_df('AB', [1, 2])
df2 = make_df('AB', [3, 4])
print(display('df1', 'df2', 'pd.concat([df1, df2])'))
