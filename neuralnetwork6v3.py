# same as neuralnetwork6
# train with datasets 5, 6 and 7, test on 18 only
# use LSTM model
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# pklfiles = ['frames.pkl', 'frames2.pkl', 'frames3.pkl']
pklfiles = ['test2.pkl', 'test3.pkl', 'test4.pkl', 'test1.pkl']
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
df10 = frames[3][0]
df11 = frames[3][1]
df12 = frames[3][2]

frameset1 = [df1, df2, df3]
frameset2 = [df4, df5, df6]
frameset3 = [df7, df8, df9]
frameset4 = [df10, df11, df12]

allframes = [frameset1, frameset2, frameset3, frameset4]
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

# remove NaNs from the whole dataset:
print('Shape before:', dfcomb_final.shape)

print('No. NaN values: ', dfcomb_final.isnull().sum().sum())

dfcomb_final = dfcomb_final.dropna(axis=0, how='any')

print('Shape after:', dfcomb_final.shape)

# saving dataframe to use to plot at end of script:
with open('dfcomb_final.pkl', 'wb') as handle:
    pickle.dump(dfcomb_final, handle, protocol=pickle.HIGHEST_PROTOCOL)



# normalising data:
scaler = MinMaxScaler()

col_names = list(dfcomb_final.columns)
row_num = list(dfcomb_final.index)

dfcomb_final_scaled = scaler.fit_transform(dfcomb_final.to_numpy())
dfcomb_final = pd.DataFrame(dfcomb_final_scaled, columns=col_names, index=row_num)





#                                               get the x and y data:

# get three different dataframes and randomise order of rows
df_training = dfcomb_final[:-167]
df_test = dfcomb_final.tail(167)


df_training = df_training.sample(frac=1, random_state=22)
df_test = df_test.sample(frac=1, random_state=22)


# X and y data split for testing and training

X_train = df_training.drop('Average SOH', axis=1)
y_train = df_training['Average SOH']

X_test = df_test.drop('Average SOH', axis=1)
y_test = df_test['Average SOH']


print('yeee')


# ------------------------- Neural network ------------------------- #
torch.manual_seed(0)

class MyModule(nn.Module):
    # Initialize the parameters
    def __init__(self, num_inputs, num_outputs, hidden_size, num_layers):
        super(MyModule, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size


        self.lstm = nn.LSTM(num_inputs, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, num_outputs)
        self.activation = nn.ReLU()
        # self.linear = nn.Linear(hidden_size, num_outputs)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, num_outputs)



    # Forward pass
    def forward(self, input):
        h_0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size)  # hidden state
        c_0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size)  # internal state

        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(input, (h_0, c_0))  # lstm with input, hidden, and internal state

        pred = self.fc1(output[:, -1, :])

        pred = self.dropout(pred)


        pred = self.activation(pred)


        pred = self.fc2(output[:, -1, :])

        # pred = self.linear(pred)

        return pred


# Instantiate the custom module
# 6 inputs (from the features), one output (SOH) and hidden size is 19 neurons
model = MyModule(num_inputs=6, num_outputs=1, hidden_size=100, num_layers=1)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.

# criterion = torch.nn.MSELoss(size_average=False)
loss_fn = torch.nn.MSELoss()

# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# convert to pytorch tensors:

# convert X_train and X_test to numpy arrays

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)



# adding another dimension for LSTM model:
X_train_tensor = torch.reshape(X_train_tensor,   (X_train_tensor.shape[0], 1, X_train_tensor.shape[1]))

X_test_tensor = torch.reshape(X_test_tensor,  (X_test_tensor.shape[0], 1, X_test_tensor.shape[1]))



# training and test data
train_dataloader = DataLoader(list(zip(X_train_tensor, y_train_tensor)), batch_size=100)
# train_dataloader = DataLoader(list(zip(X_train_tensor, y_train_tensor)), batch_size=32, shuffle=True)


test_dataloader = DataLoader(list(zip(X_test_tensor, y_test_tensor)), batch_size=100)
# test_dataloader = DataLoader(list(zip(X_test_tensor, y_test_tensor)), batch_size=32, shuffle=True)


# Check it's working
for batch, (X, y) in enumerate(train_dataloader):
    print(f"Batch: {batch+1}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    break


# Training model test:
num_epochs = 400
training_losses = []
validation_losses = []

val_results = []

for epoch in range(num_epochs):
    batch_loss = []
    # training losses:
    for X, y in train_dataloader:
        model.train()
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

    with torch.no_grad():
        val_losses = []
        model.eval()
        for X, y in test_dataloader:
            # model.eval()

            outputs = model(X)
            val_results.append(outputs.numpy())
            val_loss = loss_fn(outputs, y)
            val_losses.append(val_loss.item())
        validation_loss = np.mean(val_losses)
        validation_losses.append(validation_loss)

    val_results = np.concatenate(val_results, axis=0)

    print(f"[{epoch+1}] Training loss: {training_loss:.3f}\t Validation loss: {validation_loss:.3f}")

plt.figure(1)
plt.plot(training_losses, label='Training Loss')
plt.plot(validation_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


# ------------------------- Plot predicted values against cycle number ------------------------- #

# get the inverse scaler first:

# Get the scaling range for the final column of dfcomb_final
soh_range = scaler.data_range_[-1]
soh_min = scaler.data_min_[-1]

# Unnormalize val_results
unnormalized_val_results = val_results * soh_range + soh_min



print(X_test.index, unnormalized_val_results)
plt.figure(2)
# plt.plot(X_test.index, val_results, label='Predicted SOH')
plt.scatter(X_test.index, unnormalized_val_results, label='Predicted SOH')
plt.xlabel('Cycle Number')
plt.ylabel('SOH')
plt.legend()


# plot true results on same graph:
with open('dfcomb_final.pkl', 'rb') as handle:
    dfcomb_final = pickle.load(handle)

# # get results back in correct order
# df_test = dfcomb_final.tail(167)
#
# # rewriting X_test and y_test to get original data before shuffling
# X_test = df_test.drop('Average SOH', axis=1)
# y_test = df_test['Average SOH']

# .tail(167)

# plt.scatter(dfcomb_final.index, dfcomb_final['Average SOH'], label='True SOH')
plt.scatter(dfcomb_final.tail(167).index, dfcomb_final.tail(167)['Average SOH'], label='True SOH')

plt.legend()

plt.show()

print('hello')