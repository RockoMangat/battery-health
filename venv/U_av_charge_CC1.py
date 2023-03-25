# Finding average voltage of CC CHARGING process

import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

# import dataframes
from pandas6 import load_df
dfs = load_df()

# Nearest value function
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# number of dataset using - 0,1,2 for 3 battery datasets
dataset = [0,1,2]


# calling first dataframe/dataset
a = dataset[0]
x = dfs[a]
print(x)


# create new empty lists which data will be added to from main dictionary, for graphs
time = []
charge_CC_voltage = []
av = []
chargetime = []
fixedtime = []

# loop to create graph:
for i, column in x.items():
    # i is the discharge cycle number
    print('i: ', i)

    # print('Column 7 (time): ', column[7])

    # print('Column 2 (charge voltage): ', column[2])

    time.append(column[7])

    charge_CC_voltage.append(column[2])

    # print('len before', len(charge_CC_voltage[i]))


    # apply below if above 4.2V and remove these values from charge_CC_voltage and time lists - need to create loop to eliminate not just first value
    # result = next(k for k, value in enumerate(charge_CC_voltage[i]) if value > 4.2 or value < 1)
    # del charge_CC_voltage[i][result]
    # del time[i][result]

    # print('len after', len(charge_CC_voltage[i]))


    # finding the average voltage for each line before it reaches 4.2V
    nu = [n for n in charge_CC_voltage[i] if n < 4.2]
    # print('Voltages for current cycle: ', nu)

    test = charge_CC_voltage[i]
    # average voltage for current cycle
    av.append(sum(nu)/len(nu))
    print('Average voltage for current cycle: ', av[i])


    # time taken to charge: find index of the first value which reaches 4.2V
    result = next(k for k, value in enumerate(charge_CC_voltage[i]) if value > 4.2)
    # print(charge_CC_voltage[i][result])

    # print('Time taken to reach full charge: ', time[i][result])
    chargetime.append(time[i][result])


    # voltage increment in fixed time: find index of value around generic time of 1000s
    # print('Closest time: ', time[i][find_nearest(time[i], value = 1000)])
    # print('Closest voltage: ', charge_CC_voltage[i][find_nearest(time[i], value = 1000)])
    fixedtime.append(charge_CC_voltage[i][find_nearest(time[i], value = 1000)])





    # plot graph
    plt.plot(time[i], charge_CC_voltage[i])

# average voltage of CC charge process
u_av = sum(av)/len(av)
print('Average voltage for all cycles (charge): ', u_av)

# average time taken to charge
t_av = sum(chargetime)/len(chargetime)
print('Average time for all cycles (charge): ', t_av)

# Average voltage increment in fixed time
deltau_av = sum(fixedtime)/len(fixedtime)
print('Average voltage increment in fixed time: ', deltau_av)


plt.plot(time[i], charge_CC_voltage[i])
plt.xlabel('Time (s)')
plt.ylabel('Average voltage of CC charge process (V)')
plt.xlim(0, 3500)
plt.ylim(3.4, 4.3)
plt.show()