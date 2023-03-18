# Using pandas to create dataframe for cleandata6test.py - the chatgpt file which creates one big dictionary containing the data for all three sets of battery data

import pandas as pd
import pickle
import matplotlib.pyplot as plt

# import data from other file
from cleandata6test import all_discharge, all_charge, all_impedance

# number of dataset using - 0,1,2 for 3 battery datasets
num = 0

# set up dataframe for each discharge dataset:
df = pd.DataFrame.from_dict(all_discharge[num])

print(df)


print('Number of discharge cycles: ', len(all_discharge[num]))
range = range(1,len(all_discharge[num])+1)


# create new empty lists which data will be added to from main dictionary
charge_cycle = []
capacity = []

for i, column in df.items():
    print('i: ', i)
    # print('column 1: ', column[1])
    # print('column 2: ', column[2])
    print('column 8 (capacity): ', column[8])

    charge_cycle.append(i)
    capacity.append(column[8])


plt.plot(range, capacity)
plt.xlabel('Number of discharge cycles')
plt.ylabel('Capacity (Ah)')
plt.show()

