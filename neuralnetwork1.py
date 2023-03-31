import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn

from sklearn.model_selection import train_test_split

# import dataframes
from pandas5v2 import load_df
dfs = load_df()

def nn1(X,Y):

    X = np.arange(20).reshape(-1, 1)
    Y = np.array([5, 12, 11, 19, 30, 29, 23, 40, 51, 54, 74, 62, 68, 73, 89, 84, 89, 101, 99, 106])

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    X_train.shape, X_test.shape, Y_train.shape, Y_test.shape






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
