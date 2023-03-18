import pandas as pd
import pickle
import matplotlib.pyplot as plt

# import data from other file
# from cleandata4 import discharge, charge, impedance
from cleandata3 import discharge, charge, impedance

df = pd.DataFrame.from_dict(discharge)
# df = pd.DataFrame(discharge)

for key, value in discharge.items():
    print('key: ', key)
    print('value: ', value)

print(df)
# print(df.info())
# print(df.shape)

# df.plot(kind='scatter', x='date_time', y='current_load')

# print column values - that is no. cycles
print('Column names:', df.columns)

# print row labels
print('Row names:', df.index)

# print('Current battery data:', df.)

print('Number of discharge cycles: ', len(discharge))

# print(discharge[])



# df.plot(kind='scatter', x=df.columns, y='capacity')
df.plot(x=range(len(discharge)), y='capacity')
# plt.show()


# Use pickle:

# Serialisation:
with open('test.pickle', 'wb') as f:
    pickle.dump(df, f)

# Deserialisation:
with open('test.pickle', 'rb') as f:
    test_dict_reconstructed = pickle.load(f)