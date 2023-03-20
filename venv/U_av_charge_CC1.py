# Finding average voltage of CC CHARGING process

import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

# import dataframes
from pandas6 import load_df
dfs = load_df()

# plt.close()

# list of DataFrame names
df_names = ['df1', 'df2', 'df3']

# loop through the list of DataFrame names
# for name in df_names:

# calling first dataframe
a = df_names[0]
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

    # plot graph
    plt.plot(time[i], charge_CC_voltage[i])



plt.plot(time[i], charge_CC_voltage[i])
plt.xlabel('Time (s)')
plt.ylabel('Average voltage of CC charge process (V)')
plt.show()