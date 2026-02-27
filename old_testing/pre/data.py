import numpy as np
import csv

with open('./data/recipes.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader) 
    data = np.array(list(reader), dtype=str)

print(data.shape)

X = data[:, 7:8]
y = data[:, -1]
print(X)
# print(y.shape)