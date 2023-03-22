# Finding average voltage of CC CHARGING process

import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

# import dataframes
from pandas6 import load_df
dfs = load_df()

# plt.close()

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

# loop to create graph:
for i, column in x.items():
    # i is the discharge cycle number
    print('i: ', i)

    print('Column 7 (time): ', column[7])

    print('Column 2 (charge voltage): ', column[2])

    time.append(column[7])

    charge_CC_voltage.append(column[2])


    # apply below if above 4.2V and remove these values from charge_CC_voltage and time lists
    result = next(k for k, value in enumerate(charge_CC_voltage[i]) if value > 4.2 or value < 1)
    del charge_CC_voltage[i][result]
    del time[i][result]

    # finding the average voltage for each line before it reaches 4.2V
    nu = [n for n in charge_CC_voltage[i] if n < 4.2]
    print('Voltages for current cycle: ', nu)

    test = charge_CC_voltage[i]
    # average voltage for current cycle
    av.append(sum(nu)/len(nu))
    print('Average voltage for current cycle: ',av[i])


    # plot graph
    plt.plot(time[i], charge_CC_voltage[i])

# average voltage of CC charge process
u_av = sum(av)/len(av)
print('Average voltage for all cycles (charge): ', u_av)

plt.plot(time[i], charge_CC_voltage[i])
plt.xlabel('Time (s)')
plt.ylabel('Average voltage of CC charge process (V)')
plt.xlim(0, 3500)
plt.ylim(3.4, 4.3)
plt.show()