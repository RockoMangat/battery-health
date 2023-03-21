# finding the SOH values from capacity data:

import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

# import dataframes
from pandas5v2 import load_df
dfs = load_df()


# number of dataset using - 0,1,2 for 3 battery datasets
dataset = [0,1,2]



# calling first dataframe
a = dataset[0]
b = dataset[1]
c = dataset[2]

# x = B0005, y = B0006, z = B0018
x = dfs[a]
y = dfs[b]
z = dfs[c]
print(x)


# create new empty lists which data will be added to from main dictionary, for graphs
charge_cycle = []
capacity = []
soh = []
fullcapacity = 2

# loop to create graph:
for i, column in y.items():
    # added in the below to ensure it prints only when script run directly
    if __name__ == '__main__':
        print('i: ', i)

        print('Column 8 (capacity): ', column[8])

        charge_cycle.append(i)
        capacity.append(column[8])

        l = column[8]
        print(l[0])
        soh.append(l[0] / fullcapacity)


range = charge_cycle

print(soh)

ax = plt.plot(range,soh)

plt.xlabel('Cycle')
plt.ylabel('SOH (%)')

plt.show()