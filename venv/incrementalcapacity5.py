# Incremental capacity graphs:
# using discharge data
#  same as incrementalcapacity4, but finding delta_u1 and delta_u2 for all cycles and implementing normalisation too

import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from sklearn.preprocessing import MinMaxScaler

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
# cycles_to_loop = [0, 72, 84, 95, 107, 119, 131, 143, 155, 167]
# number of charge cycles:
no_cc = len(x.columns)

# Cycle range
cycles_to_loop = list(range(0, no_cc))

# cyles below first
final_cycles = []

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
        # print(discharge_CC_voltage)
        # ax = plt.plot(discharge_CC_voltage[n][1:], inc_cap_smoothed[n], label=round(soh[n],2))



        # max ICA for current cycle:
        maxica.append(max(inc_cap_smoothed[n]))
        print('n: ', n)

        # plots graph only if it has an ICA lower than the first cycle

        if n == 0:
            ax = plt.plot(discharge_CC_voltage[n][1:], inc_cap_smoothed[n], label=round(soh[n],2))
            plt.legend()
            print('space')
            continue


        if maxica[n] > maxica[0]:
            continue

        else:
            ax = plt.plot(discharge_CC_voltage[n][1:], inc_cap_smoothed[n], label=round(soh[n], 2))
            plt.legend()

            final_cycles.append(n)




# plt.plot(discharge_CC_voltage[i], inc_cap[i])
plt.xlabel('Terminal Voltage (V)')
plt.ylabel('Incremental Capacity (Ah/V)')
plt.legend()
# plt.show()


# find max ICA value and terminal voltage for second cycle:
print('Max ICA of second graph: ', maxica[1])

# Initialise delta_u1 and delta_u2 and nothing else - since everything else in for loop is just used to calculate delta and can be refreshed each cycle
delta_u1 = []
delta_u2 = []

# no = list(range(0, 10))

# Loop through cycles to get centres for each cycle
for cycle in cycles_to_loop:
    # skip past cycle 0 - we want difference in voltage between centre of cycle 0 and all other cycles
    if cycle == 0:
        continue
    # index of max ICA within that cycle
    maxica_index0 = next((i for i, j in enumerate(inc_cap_smoothed[cycle]) if j == maxica[cycle]), None)
    centre = discharge_CC_voltage[cycle][maxica_index0]
    print('Terminal Voltage 0: ', centre)

    # Find indexes of value in first cycle, that reach max ICA of current cycle
    nearest1 = find_nearest(inc_cap_smoothed[0], maxica[cycle])

    # array specifically for this data and updating it:
    arrayrt = list(inc_cap_smoothed[0])
    # delete the values because we want to find where it coincides on the other side too - find_nearest would go back to the same value as before otherwise
    arrayrt.remove(nearest1)

    # Find indexes of value in same cycle, but other side of graph, that reaches max ICA of current cycle
    nearest2 = find_nearest(arrayrt, maxica[cycle])
    # print(inc_cap_smoothed[0])

    # Index and terminal voltages for the two values
    maxica_index1 = next((i for i, j in enumerate(inc_cap_smoothed[0]) if j == nearest1), None)
    maxica_index2 = next((i for i, j in enumerate(inc_cap_smoothed[0]) if j == nearest2), None)

    # print('Index 1: ', maxica_index1)
    # print('Index 2: ', maxica_index2)

    # print('ICA Value 1: ', inc_cap_smoothed[0][maxica_index1])
    # print('ICA Value 2: ', inc_cap_smoothed[0][maxica_index2])

    t_voltage1 = discharge_CC_voltage[0][maxica_index1]
    t_voltage2 = discharge_CC_voltage[0][maxica_index2]
    # print('Terminal Voltage 1: ', discharge_CC_voltage[0][maxica_index1])
    # print('Terminal Voltage 2: ', discharge_CC_voltage[0][maxica_index2])

    # Final differences
    delta_u1.append( abs((centre - t_voltage1)) )
    delta_u2.append( abs(centre - t_voltage2) )


print(delta_u1)
print(delta_u2)

# ------------------------ NORMALISING DATA ------------------------ #

# number of charge cycles:
# no_cc = len(x.columns)

# Cycle range
# cc = list(range(0, no_cc))
cycles_to_loop.remove(0)

# Create an instance of the scaler
scaler = MinMaxScaler()

# Convert to numpy array
delta_u1_np = np.array(delta_u1)
delta_u2_np = np.array(delta_u2)

# Reshape the array to have two dimensions
delta_u1_np = delta_u1_np.reshape(-1, 1)
delta_u2_np = delta_u2_np.reshape(-1, 1)

# Normalize the data
normalized_data1 = scaler.fit_transform(delta_u1_np)
normalized_data2 = scaler.fit_transform(delta_u2_np)

# Convert back to list, to plot
nd1 = normalized_data1.tolist()
plt.figure(2)
plt.plot(cycles_to_loop, nd1)
plt.xlabel('Cycle')
plt.ylabel('Normalised Feature 5')
# plt.show()


nd2 = normalized_data2.tolist()
plt.figure(3)
plt.plot(cycles_to_loop, nd2)
plt.xlabel('Cycle')
plt.ylabel('Normalised Feature 6')
plt.show()






