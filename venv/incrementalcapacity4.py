# Incremental capacity graphs:
# using discharge data
#  same as incrementalcapacity3, greater sigma and finding values for du1 and du2

import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

# import dataframes
from pandas5v2 import load_df
dfs = load_df()

# Nearest value function
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# number of dataset using - 0,1,2 for 3 battery datasets
dataset = [0, 1, 2]

# calling first dataframe
a = dataset[0]
x = dfs[a]

# create new empty lists which data will be added to from main dictionary, for graphs
discharge_CC_voltage = []
dv = []
inc_cap = {}
inc_cap_smoothed ={}
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
cycles_to_loop = [0, 72, 84, 95, 107, 119, 131, 143, 155, 167]


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

        inc_cap_smoothed[i] = {}

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
        inc_cap_smoothed[n] = scipy.ndimage.gaussian_filter1d(list(inc_cap[i].values()), sigma=4)


        # plot of smoothed data for chosen cycles
        print(discharge_CC_voltage)
        # ax = plt.plot(discharge_CC_voltage[n][1:], inc_cap_smoothed, label = i)
        ax = plt.plot(discharge_CC_voltage[n][1:], inc_cap_smoothed[n], label=round(soh[n],2))
        plt.legend()
        print('space')


        # max ICA for current cycle:

        maxica.append(max(inc_cap_smoothed[n]))
        print('n: ', n)



# plt.plot(discharge_CC_voltage[i], inc_cap[i])
plt.xlabel('Terminal Voltage (V)')
plt.ylabel('Incremental Capacity (Ah/V)')
plt.legend()
plt.show()



# find max ICA value and terminal voltage for second cycle:
print('Max ICA of second graph: ', maxica[1])

maxica_index0 = next((i for i, j in enumerate(inc_cap_smoothed[1]) if j == maxica[1]), None)
centre = discharge_CC_voltage[1][maxica_index0]
print('Terminal Voltage 0: ', centre)

# print('Index: ', maxica_index)
# if maxica[1] == inc_cap_smoothed[1][maxica_index]:
#     print('Works')
    # print('Value check: ', inc_cap_smoothed[1][aaa])


# Find indexes of value in first cycle, that reach max ICA of next graph, of 4.262
nearest1 = find_nearest(inc_cap_smoothed[0], maxica[1])

# array specifically for this data and updating it:
arrayrt = list(inc_cap_smoothed[0])
arrayrt.remove(nearest1)

# Find indexes of value in second cycle, that reach max ICA of next graph, of 4.262
nearest2 = find_nearest(arrayrt, maxica[1])
print(inc_cap_smoothed[0])

# Index and terminal voltages for the two values
maxica_index1 = next((i for i, j in enumerate(inc_cap_smoothed[0]) if j == nearest1), None)
maxica_index2 = next((i for i, j in enumerate(inc_cap_smoothed[0]) if j == nearest2), None)

print('Index 1: ', maxica_index1)
print('Index 2: ', maxica_index2)

print('ICA Value 1: ', inc_cap_smoothed[0][maxica_index1])
print('ICA Value 2: ', inc_cap_smoothed[0][maxica_index2])

t_voltage1 = discharge_CC_voltage[0][maxica_index1]
t_voltage2 = discharge_CC_voltage[0][maxica_index2]
print('Terminal Voltage 1: ', discharge_CC_voltage[0][maxica_index1])
print('Terminal Voltage 2: ', discharge_CC_voltage[0][maxica_index2])

# Final differences
delta_u1 = -(centre - t_voltage1)
delta_u2 = centre - t_voltage2

print(delta_u1)
print(delta_u2)









