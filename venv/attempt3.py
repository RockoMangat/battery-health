import numpy as np
import pickle
import scipy.io as sio


# load .mat file
mat_data = sio.loadmat('/Users/rohanmangat/Downloads/5. Battery Data Set/1. BatteryAgingARC-FY08Q4/B0005.mat')


# only extract data needed from file:
data_needed = {}
for key in mat_data.keys():
    if not key.startswith('__'):
        data_needed[key] = mat_data[key]


# Use pickle:

# Serialisation:
with open('test.pickle', 'wb') as f:
    pickle.dump(data_needed, f)

# Deserialisation:
with open('test.pickle', 'rb') as f:
    test_dict_reconstructed = pickle.load(f)


# print all the keys
# print('All keys: ', test_dict_reconstructed.keys())

# print all values - if you use this, it will end up showing all values in cleandata script too as it runs rest of script
print(test_dict_reconstructed.values())

