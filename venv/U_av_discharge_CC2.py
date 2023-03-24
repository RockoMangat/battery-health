# Finding average voltage of CC DISCHARGING process
# second version - removing all values after min value

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
x = dfs[a]
print(x)


# create new empty lists which data will be added to from main dictionary, for graphs
time = []
discharge_CC_voltage = []
av = []

# loop to create graph:
for i, column in x.items():
    # i is the discharge cycle number
    print('i: ', i)

    print('Column 7 (time): ', column[7])

    print('Column 2 (discharge voltage): ', column[2])

    time.append(column[7])

    discharge_CC_voltage.append(column[2])

    # find min value
    minval = min(discharge_CC_voltage[i])
    # print(minval)
    # min value index
    minindex = discharge_CC_voltage[i].index(minval)
    # remove all values after minvalue
    del discharge_CC_voltage[i][minindex+1:]
    del time[i][minindex+1:]

    # finding the average voltage for each line
    nu = discharge_CC_voltage[i]

    # checking if updated and old values removed
    test = discharge_CC_voltage[i]
    # average voltage for current cycle
    av.append(sum(nu) / len(nu))
    print('Average voltage for current cycle: ', av[i])


    # plot graph
    plt.plot(time[i], discharge_CC_voltage[i])

    print('tester')

# average voltage of CC discharge process
u_av = sum(av) / len(av)
print('Average voltage for all cycles (discharge): ', u_av)


plt.plot(time[i], discharge_CC_voltage[i])
plt.xlabel('Time (s)')
plt.ylabel('Average voltage of CC discharge process (V)')
plt.show()