import numpy as np
import pickle
import scipy.io as sio

# open, serialise, deserialise .mat files

# load .mat file
mat_data = sio.loadmat('/Users/rohanmangat/Downloads/5. Battery Data Set/1. BatteryAgingARC-FY08Q4/B0005.mat')

# convert .mat file to python dictionary
#data_dict = {}
#for key in mat_data.keys():
    #data_dict[key] = mat_data[key]

# use pickle:

# Serialisation:
with open('test.pickle', 'wb') as f:
    pickle.dump(data_dict,f)
#print("Written object", mat_data)


# Deserialisation:
with open('test.pickle', 'rb') as infile:
    test_dict_reconstructed = pickle.load(infile)
    print('Array shape: ' + str(test_dict_reconstructed.shape))
    print('Data type: ' +str(type(test_dict_reconstructed)))


#print(test_dict_reconstructed)


print('mat_data type:', type(mat_data))
print('test_duct_reconstructed:', type(test_dict_reconstructed))


if mat_data == test_dict_reconstructed:
    print("good??")


if isinstance(test_dict_reconstructed, np.ndarray):
    print("Shape:", data.shape)
    print("Data type:", data.dtype)
else:
    print("Object is not a NumPy array.")