# Finding average voltage of CC DISCHARGING process
# second version - removing all values after min value

import matplotlib.pyplot as plt

# import dataframes
from pandas5v2 import load_df
dfs = load_df()

from sohdischarge import ab, charge_cycle

def discharge_data(dataset):

    x = dfs[dataset]
    print(x)

    # number of charge cycles:
    no_cc = len(x.columns)

    # Cycle range
    cc = list(range(0, no_cc))

    # create new empty lists which data will be added to from main dictionary, for graphs
    time = []
    discharge_CC_voltage1 = []
    av = []

    # loop to create graph:
    for i, column in x.items():
        # i is the discharge cycle number
        print('i: ', i)

        print('Column 7 (time): ', column[7])

        print('Column 2 (discharge voltage): ', column[2])

        time.append(column[7])

        discharge_CC_voltage1.append(column[2])

        # find min value
        minval = min(discharge_CC_voltage1[i])
        # print(minval)
        # min value index
        minindex = discharge_CC_voltage1[i].index(minval)
        # remove all values after minvalue
        del discharge_CC_voltage1[i][minindex+1:]
        del time[i][minindex+1:]

        # finding the average voltage for each line
        nu = discharge_CC_voltage1[i]

        # checking if updated and old values removed
        test = discharge_CC_voltage1[i]
        # average voltage for current cycle
        av.append(sum(nu) / len(nu))
        print('Average voltage for current cycle: ', av[i])


        # plot graph
        plt.plot(time[i], discharge_CC_voltage1[i])

        print('tester')


    plt.plot(time[i], discharge_CC_voltage1[i])
    plt.xlabel('Time (s)')
    plt.ylabel('Average voltage of CC discharge process (V)')
    plt.show()

    return av, cc, ab, charge_cycle