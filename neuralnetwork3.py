# combining all three datasets into one
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

pklfiles = ['frames.pkl', 'frames2.pkl', 'frames3.pkl']
frames = []

for filename in pklfiles:
    # import dataframes from main
    with open(filename, 'rb') as handle:
        frames1 = pickle.load(handle)

    frames.append(frames1)

# ------------------------- Sorting data input ------------------------- #
df1 = frames[0][0]
df2 = frames[0][1]
df3 = frames[0][2]
df4 = frames[1][0]
df5 = frames[1][1]
df6 = frames[1][2]
df7 = frames[2][0]
df8 = frames[2][1]
df9 = frames[2][2]

frameset1 = [df1, df2, df3]
frameset2 = [df4, df5, df6]
frameset3 = [df7, df8, df9]

allframes = [frameset1, frameset2, frameset3]
dfcomb_final = pd.DataFrame()

# combining all dataframes
for frameset in allframes:
    dfcomb = pd.concat(frameset, axis=1)
    print(dfcomb.shape)

    dfcomb = dfcomb.drop(['SOH charge cycles', 'SOH discharge cycles', 'SOH discharge cycles 2', 'SOH discharge 2'], axis=1)
    print(dfcomb.shape)

    # fixing issue of value stored as lists:
    dfcomb = dfcomb.applymap(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)

    # find average SOH for charge and discharge, then remove those 2 columns
    dfcomb['Average SOH'] = dfcomb[['SOH charge', 'SOH discharge']].mean(axis=1)
    dfcomb = dfcomb.drop(['SOH charge', 'SOH discharge'], axis=1)

    dfcomb_final = pd.concat([dfcomb, dfcomb_final], axis=0)


# get the x and y data:
X = dfcomb_final.drop('Average SOH', axis=1)
print(X.shape)

y = dfcomb_final['Average SOH']
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
model = MyModule(num_inputs=6, num_outputs=1, hidden_size=19)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.

# criterion = torch.nn.MSELoss(size_average=False)
loss_fn = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# convert to pytorch tensors:

# convert X_train and X_test to numpy arrays
# X_train_np = X_train.to_numpy(dtype=np.float32)
# X_test_np = X_test.to_numpy(dtype=np.float32)

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)

# training and test data
# train_data = (X_train_tensor, y_train_tensor)
train_dataloader = DataLoader(list(zip(X_train_tensor, y_train_tensor)), batch_size=32)

# test_data = (X_test_tensor, y_test_tensor)
test_dataloader = DataLoader(list(zip(X_test_tensor, y_test_tensor)), batch_size=32)

# Check it's working
for batch, (X, y) in enumerate(train_dataloader):
    print(f"Batch: {batch+1}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    break


# Training model test:
num_epochs = 700
training_losses = []
validation_losses = []

val_results = []

for epoch in range(num_epochs):
    batch_loss = []
    # training losses:
    for X, y in train_dataloader:
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        batch_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    training_loss = np.mean(batch_loss)
    training_losses.append(training_loss)

    # Validation:

    val_results = []
    cycle_numbers = []

    with torch.no_grad():
        val_losses = []
        model.eval()
        for X, y in test_dataloader:
            outputs = model(X)
            val_results.append(outputs.numpy())
            cycle_numbers.append(X[:, 0].numpy())
            val_loss = loss_fn(outputs, y)
            val_losses.append(val_loss.item())
        validation_loss = np.mean(val_losses)
        validation_losses.append(validation_loss)

    val_results = np.concatenate(val_results, axis=0)
    cycle_numbers = np.concatenate(cycle_numbers, axis=0)

    print(f"[{epoch+1}] Training loss: {training_loss:.3f}\t Validation loss: {validation_loss:.3f}")

plt.figure(1)
plt.plot(training_losses, label='Training Loss')
plt.plot(validation_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.figure(2)
plt.plot(cycle_numbers, val_results, label='Predicted SOH')
plt.xlabel('Cycle Number')
plt.ylabel('Predicted SOH')
plt.show()



# plt.figure()
# test_output = outputs.detach().numpy()
# plt.plot(outputs.detach().numpy(), label='predicted')
# # plt.plot(pred.detach().numpy(), label='predicted')
# plt.plot(y_train, label='actual')
# plt.legend()
# plt.show()


# plt.figure(1)
# plt.plot(np.array(loss_values))
# plt.title("Step-wise Loss")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")



print('hello')