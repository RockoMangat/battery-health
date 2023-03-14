import numpy as np
import pickle
import scipy.io as sio


# load .mat file
mat_data = sio.loadmat('/Users/rohanmangat/Downloads/5. Battery Data Set/1. BatteryAgingARC-FY08Q4/B0005.mat')

# Use pickle:

# Serialisation:
with open('test.pickle', 'wb') as f:
    pickle.dump(mat_data, f)

# Deserialisation:
with open('test.pickle', 'rb') as f:
    test_dict_reconstructed = pickle.load(f)

# check data types:
print('mat_data type:', type(mat_data))
print('test_duct_reconstructed:', type(test_dict_reconstructed))

# print all the keys
print('All keys: ', test_dict_reconstructed.keys())

# find keys which start with '__'
for key, value in test_dict_reconstructed.items() :
    if key.startswith('__') :
        print(key)
        print(value)

# hence don't need the data from these keys
