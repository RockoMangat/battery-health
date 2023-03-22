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

# loop to create graph:
for i, column in x.items():
    # i is the discharge cycle number
    print('i: ', i)

    print('Column 7 (time): ', column[7])

    print('Column 2 (charge voltage): ', column[2])

    time.append(column[7])

    charge_CC_voltage.append(column[2])


    # apply below if above 4.2V and remove these values from charge_CC_voltage and time lists
    result = next(k for k, value in enumerate(charge_CC_voltage[i]) if value > 4.2 or value < 3)
    del charge_CC_voltage[i][result]
    del time[i][result]

    # plot graph
    plt.plot(time[i], charge_CC_voltage[i])


plt.plot(time[i], charge_CC_voltage[i])
plt.xlabel('Time (s)')
plt.ylabel('Average voltage of CC charge process (V)')
plt.xlim(0,3500)
plt.ylim(3.4,4.3)
plt.show()