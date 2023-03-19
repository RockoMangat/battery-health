# Using pandas to create dataframe for cleandata6test.py - the chatgpt file which creates one big dictionary containing the data for all three sets of battery data

import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np


# import data from other file
from cleandata6test import all_discharge, all_charge, all_impedance

# number of dataset using - 0,1,2 for 3 battery datasets
dataset = [0,1,2]

# create a list of DataFrame names
df_names = ['df1', 'df2', 'df3']

# create an empty dictionary to store the DataFrames
dfs = {}

# loop through the list of DataFrame names
for name in df_names:
    # create an empty DataFrame with the name as the key
    dfs[name] = pd.DataFrame()

    # set up dataframe for each discharge dataset
    for num in dataset:
        # add in data from discharge dataset
        dfs[name] = pd.DataFrame.from_dict(all_discharge[num])
        # checking dataframe to see if correct
        x = dfs[name]

        # rename the dataframe columns to be the number of discharge cycles
        dfs[name].columns = np.arange(len(dfs[name].columns))

        # checking dataframe to see if correct
        x = dfs[name]



        # create new empty lists which data will be added to from main dictionary, for graphs
        charge_cycle = []
        capacity = []

        # loop to create graph:
        for i, column in dfs[name].items():
            print('i: ', i)

            print('Column 8 (capacity): ', column[8])

            charge_cycle.append(i)
            capacity.append(column[8])


        # range = np.arange(len(dfs[name].columns))
        range = charge_cycle

        ax = plt.plot(range, capacity, label=name)



plt.xlabel('Number of discharge cycles')
plt.ylabel('Capacity (Ah)')

plt.legend()
plt.show()
