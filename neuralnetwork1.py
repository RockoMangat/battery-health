import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# import dataframes from main
with open('frames.pkl', 'rb') as handle:
    frames = pickle.load(handle)

# ------------------------- Sorting data input ------------------------- #
df1 = frames[0]
df2 = frames[1]
df3 = frames[2]

allframes = [df1, df2, df3]

# combining all dataframes
dfcomb = pd.concat(allframes, axis=1)
print(dfcomb.shape)

dfcomb = dfcomb.drop(['SOH charge cycles', 'SOH discharge cycles', 'SOH discharge cycles 2', 'SOH discharge 2'], axis = 1)
print(dfcomb.shape)

# find average SOH for charge and discharge, then remove those 2 columns
dfcomb['Average SOH'] = dfcomb[['SOH charge', 'SOH discharge']].mean(axis=1)
dfcomb = dfcomb.drop(['SOH charge', 'SOH discharge'], axis=1)


# get the x and y data:
X = dfcomb.drop('Average SOH', axis=1)
print(X.shape)

y = dfcomb['Average SOH']
print(y.shape)

# Split data into x and y, with 30% for testing and randomly shuffling data
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=22)

print('yeee')


# ------------------------- Neural network ------------------------- #

class MyModule (nn.Module):
    # Initialize the parameter
    def __init__(self, num_inputs, num_outputs, hidden_size):
        super(MyModule, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_outputs)

    # Forward pass
    def forward(self, input):
        lin = self.linear1(input)
        output = nn.functional.sigmoid(lin)
        pred = self.linear2(output)
        return pred

# Instantiate the custom module
# 6 inputs (from the features), one output (SOH) and hidden size is 19 neurons
my_module = MyModule(num_inputs=6, num_outputs=1, hidden_size=19)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(my_module.parameters(), lr=1e-4)


# convert to pytorch tensors:

# convert X_train and X_test to numpy arrays
X_train_np = X_train.to_numpy(dtype=np.float32)
X_test_np = X_test.to_numpy(dtype=np.float32)

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)



# train_loader = torch.utils.data.DataLoader()

# Training model test:
num_epochs = 100
# loss_values = dsf
