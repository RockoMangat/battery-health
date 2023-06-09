# Incremental capacity graphs:
# using discharge data

import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

# import dataframes
from pandas5v2 import load_df
dfs = load_df()

# number of dataset using - 0,1,2 for 3 battery datasets
dataset = [0,1,2]

# calling first dataframe
a = dataset[0]
x = dfs[a]

# create new empty lists which data will be added to from main dictionary, for graphs
discharge_CC_voltage = []
dv = []
inc_cap = {}

# initialise variables
t0 = 0
t1 = 0
dt = 0

# List of cycles to loop over
cycles_to_loop = np.linspace(0, 167, 15)
cycles_to_loop = np.round(cycles_to_loop)

# loop to create graph:
for i, column in x.items():
    # Check if cycle is in the list of cycles to loop over
    # if i in cycles_to_loop:
        # i is the discharge cycle number
        print('i (cycle no.) : ', i)

        # initialize dictionary for current cycle
        inc_cap[i] = {}

        discharge_CC_voltage.append(column[2])


        for val in range(len(column[3])):
            # time data when val=0
            if val == 0:
                dt = 0

            else:
                # time data
                t0 = column[7][val - 1]
                t1 = column[7][val]
                dt = t1 - t0

                # battery data
                v1 = column[2][val]
                v0 = column[2][val - 1]
                dv = v1 - v0
                current = -column[3][val]

                inc_cap[i][val] = dt * current / (-dv * 3600)


        # apply Gaussian filter to inc_cap for current cycle
        inc_cap_smoothed = scipy.ndimage.gaussian_filter1d(list(inc_cap[i].values()), sigma=3)
        # print(discharge_CC_voltage[i][val])
        # print(inc_cap[i][val])

        print(discharge_CC_voltage[0][1:])
        print((list(inc_cap[0].values())))
        # plt.plot(discharge_CC_voltage[i][val], inc_cap[i][val])

        # plot of raw, noisy data
        # plt.plot(discharge_CC_voltage[i][1:], list(inc_cap[i].values()), label=f"Cycle {i}")

        # plot of smoothed data for all cycles
        ax = plt.plot(discharge_CC_voltage[i][1:], inc_cap_smoothed, label = i)
        plt.legend()
        print('space')

        # plot of smoothed data for limited cycles
        # plt.plot(discharge_CC_voltage[cycles_to_loop][1:], inc_cap_smoothed)




# plt.plot(discharge_CC_voltage[i], inc_cap[i])
plt.xlabel('Terminal Voltage (V)')
plt.ylabel('Incremental Capacity (Ah/V)')
plt.legend()
plt.show()




# plotting separate graph for certain cycles:
fig2, ax2 = plt.subplots(1)
ax2.plot()