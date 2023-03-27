import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn


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
my_module = MyModule(num_inputs=6*3, num_outputs=10, hidden_size=20)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(my_module.parameters(), lr=1e-4)
