import pandas as pd
import pickle
import matplotlib.pyplot as plt

# import data from other file
# from cleandata4 import discharge, charge, impedance - using cleandata3 since it has capacity too
# cleandata5test does it specifically for B0005.mat / B0007.mat / B0018.mat
from cleandata5test import discharge, charge, impedance

# set up dataframe:
df = pd.DataFrame.from_dict(discharge)

print(df)


print('Number of discharge cycles: ', len(discharge))
range = range(1,len(discharge)+1)


# create new empty lists which data will be added to from main dictionary
charge_cycle = []
capacity = []

for i, column in df.items():
    print('i: ', i)
    # print('column 1: ', column[1])
    # print('column 2: ', column[2])
    print('column 8 (capacity): ', column[8])

    charge_cycle.append(i)
    capacity.append(column[8])


plt.plot(range, capacity)
plt.xlabel('Number of discharge cycles')
plt.ylabel('Capacity (Ah)')
plt.show()









# print(discharge.keys())
# print(discharge.values())

# new_dict = {}
# for key, value in discharge.items():
#     # print('key: ', key)
#     # print('value: ', value)
#
#     for key2, value2 in value.items():
#         print('key2: ', key2)
#         print('value2: ', value2)
#
#         if key2 == 'amb_temp':
#             print(type(value2))
#             # append since it is a str type
#             # new_dict['amb_temp'].append(value2)
#
#         if key2 == 'date_time':
#             new_dict['date_time'] = value2.tolist()
#
#         if key2 == 'voltage_battery':
#             new_dict['voltage_battery'] = value2.tolist()
#
#         if key2 == 'current_battery':
#             new_dict['current_battery'] = value2.tolist()



