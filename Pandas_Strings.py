import numpy as np
import pandas as pd

x = np.array([2, 3, 5, 7, 11, 13])
print(x ** 2)

data = ['peter', 'Paul', 'MARY', 'gUIDO']
[print(x.capitalize()) for x in data]

names = pd.Series(data)
print(names.str.capitalize())
monte = pd.Series(['Graham Chapman', 'John Cleese', 'Terry Gilliam',
                   'Eric Idle', 'Terry Jones', 'Michael Palin'])
print(monte.str.lower())
print(monte.str.startswith('A'))
print(monte.str.extract('([A-Za-z]+)', expand=False))
print(monte.str.findall(r'^[^AEIOU].*[^aeiou]$'))

try:
    recipes = pd.read_csv('openrecipes.csv')
except ValueError as e:
    print("ValueError:", e)
print(recipes.shape)
print(recipes.iloc[0])

