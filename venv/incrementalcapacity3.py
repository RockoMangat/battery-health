# Incremental capacity graphs:
# using discharge data
#  same as incrementalcapacity2 but certain cyles only

import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

# import dataframes
from pandas5v2 import load_df
dfs = load_df()

# number of dataset using - 0,1,2 for 3 battery datasets
dataset = [0, 1, 2]

# calling first dataframe
a = dataset[0]
x = dfs[a]

# create new empty lists which data will be added to from main dictionary, for graphs
discharge_CC_voltage = []
dv = []
inc_cap = {}
maxica = []

# initialise variables
t0 = 0
t1 = 0
dt = 0

# SOH values
capacity = []
soh = []
fullcapacity = 2

# List of cycles to loop over and creating n variable to iterate with
# cycles_to_loop = [0, 12, 24, 36, 48, 61, 72, 84, 95, 107, 119, 131, 143, 155, 167]
cycles_to_loop = [0, 72, 84, 95, 107, 119, 131, 143, 155, 167]
# cycles_to_loop = [0, 1, 2, 3, 10, 20, 30, 40, 50, 60, 80, 95, 107, 119, 131, 143, 155, 167]
# cycles_to_loop = [0, 1, 9, 48, 88, 128, 168]
n = -1

# loop to create graph:
for i, column in x.items():
    # Check if cycle is in the list of cycles to loop over
    if i in cycles_to_loop:
        # i is the discharge cycle number
        print('i (cycle no.) : ', i)

        # Update variable n to show number of cycles
        n = n + 1

        # initialize dictionary for current cycle
        inc_cap[i] = {}

        discharge_CC_voltage.append(column[2])

        capacity.append(column[8])

        soh.append(capacity[n][0] / fullcapacity)



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


        # plot of smoothed data for chosen cycles
        print(discharge_CC_voltage)
        # ax = plt.plot(discharge_CC_voltage[n][1:], inc_cap_smoothed, label = i)
        ax = plt.plot(discharge_CC_voltage[n][1:], inc_cap_smoothed, label=round(soh[n],2))
        plt.legend()
        print('space')

#       max ICA for current cycle:
#         maxica.append(max(inc_cap_smoothed))




# plt.plot(discharge_CC_voltage[i], inc_cap[i])
plt.xlabel('Terminal Voltage (V)')
plt.ylabel('Incremental Capacity (Ah/V)')
plt.legend()
plt.show()



# find max ICA value for second cycle:
print('Max ICA of second graph: ', maxica[1])