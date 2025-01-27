# -*- coding: utf-8 -*-
"""
@author: krishna
"""
import warnings
import itertools    
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import pandas_profiling as pp
import ydata_profiling
import pylab

import sys


from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import Dense, Flatten, LSTM, Input, Dropout, BatchNormalization, GRU
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

sns.set_style('dark')

warnings.filterwarnings('ignore')


df_IR = np.array(pd.read_csv('IR_Data_withlabels_3000_sim_48_exp.csv', header = None)) # simulation and exp data
# df_IR = np.array(pd.read_csv('IR_Data_withlabels_3000_sim.csv', header = None)) # simulation data

df_Thz = np.array(pd.read_csv('Thz_data_withlabels_3000_sim_48_exp.csv', header = None)) # simulation and exp data
# df_Thz = np.array(pd.read_csv('Thz_data_withlabels_3000_sim.csv', header = None)) # simulation data

X_IR = df_IR[1:,4:]
X_Thz = df_Thz[1:,6:206]
Y=df_Thz[1:,:4] #labels - k, rhocp, thickness, refractive index

data1_nonoise=X_IR #data1
data2_nonoise=X_Thz #data2




# Generate white noise
noise1 = np.random.normal(loc=0, scale=0.0, size=data1_nonoise.shape)  # Adjust scale as desired
# Add white noise to the input data
noisy_data1 = data1_nonoise + noise1

# Generate white noise
noise2 = np.random.normal(loc=0, scale=0.0, size=data2_nonoise.shape)  # Adjust scale as desired
# Add white noise to the input data
noisy_data2 = data2_nonoise + noise2

data1=noisy_data1
data2=noisy_data2
#-------------------------------------------------------
# a2=Y[:,0]
# a2_norm = a2/np.max(a2)
# # a3_norm = a3/np.max(a3)


# #concatenating the parameters
# # gt_scale = np.concatenate([a1_norm, a2_norm, a3_norm], axis=1, out=None)
# targets = np.vstack([a2_norm]).T

#------------------------------------------------------
#acce parameter
a1 = Y[:,0]
a2 = Y[:,1]
a3 = Y[:,2]
a4 = Y[:,3]


a1_norm = a1/np.max(a1)
a2_norm = a2/np.max(a2)
a3_norm = a3/np.max(a3)
a4_norm = a4/np.max(a4)

#concatenating the parameters
# gt_scale = np.concatenate([a1_norm, a2_norm, a3_norm], axis=1, out=None)
targets = np.vstack([a1_norm, a2_norm, a3_norm, a4_norm]).T


#------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Define LSTM model with attention
class LSTMAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attention_weights = self.attention(lstm_out).squeeze(2)
        attention_weights = torch.softmax(attention_weights, dim=1)
        attended_encoding = torch.bmm(attention_weights.unsqueeze(1), lstm_out).squeeze(1)
        return attended_encoding

# # Define late-stage fusion model
# class FusionModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(FusionModel, self).__init__()
#         self.lstm1 = LSTMAttention(input_dim, hidden_dim)
#         self.lstm2 = LSTMAttention(input_dim, hidden_dim)
#         self.fc = nn.Linear(hidden_dim * 2, output_dim)

#     def forward(self, x1, x2):
#         output1 = self.lstm1(x1)
#         output2 = self.lstm2(x2)
#         fused_output = torch.cat((output1, output2), dim=1)
#         prediction = self.fc(fused_output)
#         return prediction


##############################################################################


hidden_size1= 32
hidden_size2= 32
fusion_size=64
output_size=4



class FusionModel(nn.Module):
    def __init__(self, input_size1, hidden_size1, input_size2, hidden_size2, fusion_size, output_size):
        super(FusionModel, self).__init__()

        # LSTM model 1
        self.lstm_model1 = nn.LSTM(input_size=input_size1, hidden_size=hidden_size1, batch_first=True)

        # LSTM model 2
        self.lstm_model2 = nn.LSTM(input_size=input_size2, hidden_size=hidden_size2, batch_first=True)

        # Fusion operation
        self.fusion_layer = nn.Linear(hidden_size1 + hidden_size2, fusion_size)

        # Output layer
        self.output_layer = nn.Linear(fusion_size, output_size)

    def forward(self, input1, input2):
        # Process input1 through LSTM model 1
        output1, _ = self.lstm_model1(input1)

        # Process input2 through LSTM model 2
        output2, _ = self.lstm_model2(input2)

        # Concatenate the outputs from both LSTM models
        # fusion_output = torch.cat((output1[:, -1, :], output2[:, -1, :]), dim=1)
        fusion_output = torch.cat((output1, output2), dim=1)

        # Perform fusion operation
        fusion_output = self.fusion_layer(fusion_output)

        # Predict the output
        output = self.output_layer(fusion_output)

        return output



class PropertyDataset(Dataset):
    def __init__(self, data1, data2, targets):
        self.data1 = data1
        self.data2 = data2
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        x1 = self.data1[index]
        x2 = self.data2[index]
        y = self.targets[index]
        return x1, x2, y

#-----------------------------------------------------------------------------------------
# # Example usage


# Convert data and targets to float type
data1x=np.array(data1)
data1 = data1x.astype(float)
data2x=np.array(data1)
data2 = data2x.astype(float)
targets=np.array(targets)
targets = targets.astype(float)

# Assuming you have prepared your data as 'data1', 'data2', and 'targets' arrays

# Split into training and validation sets
train_size = int(0.8 * len(data1))
train_data1 = data1[:train_size]
train_data2 = data2[:train_size]
train_targets = targets[:train_size]
val_data1 = data1[train_size:]
val_data2 = data2[train_size:]
val_targets = targets[train_size:]

# Convert data and targets to tensors
train_data1 = torch.tensor(train_data1).float()
train_data2 = torch.tensor(train_data2).float()
train_targets = torch.tensor(train_targets).float()
val_data1 = torch.tensor(val_data1).float()
val_data2 = torch.tensor(val_data2).float()
val_targets = torch.tensor(val_targets).float()

# Create dataset and dataloaders
train_dataset = PropertyDataset(train_data1, train_data2, train_targets)
val_dataset = PropertyDataset(val_data1, val_data2, val_targets)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model initialization
input_dim = data1.shape[-1]  # Dimension of input data for each LSTM model
hidden_dim = 32
output_dim = 4  # Number of properties to predict

# fusion_model = FusionModel(input_dim, hidden_dim, output_dim)
input_size1=200
input_size2=200
fusion_model = FusionModel(input_size1, hidden_size1, input_size2, hidden_size2, fusion_size, output_size)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(fusion_model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fusion_model.to(device)


# Get targets and predictions for the validation set
val_targets = []
val_predictions = []

# Define lists to store targets and predicted values
all_targets = []
all_predictions = []

# Define lists to store loss values
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    fusion_model.train()
    train_loss = 0.0

    for data1, data2, targets in train_dataloader:
        data1 = data1.to(device)
        data2 = data2.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass through fusion model
        outputs = fusion_model(data1, data2)
        predictions = outputs.squeeze()  # Assuming the predictions are in the shape (batch_size, )
        
        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Store targets and predicted values
        all_targets.extend(targets.numpy())
        all_predictions.extend(predictions.detach().numpy())
        train_loss += loss.item() * data1.size(0)

    # Compute average training loss for the epoch
    train_loss /= len(train_dataset)
    train_losses.append(train_loss)

    # Validation loop
    fusion_model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for data1, data2, targets in val_dataloader:
            data1 = data1.to(device)
            data2 = data2.to(device)
            targets = targets.to(device)

            # Forward pass through fusion model
            outputs = fusion_model(data1, data2)

            # Compute loss
            loss = criterion(outputs, targets)

            val_loss += loss.item() * data1.size(0)
            predictions = outputs.squeeze().tolist()
            val_targets.extend(targets.tolist())
            val_predictions.extend(predictions)

    # Compute average validation loss for the epoch
    val_loss /= len(val_dataset)
    val_losses.append(val_loss)
    # Print epoch statistics
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# Save the trained fusion model
torch.save(fusion_model, 'fusion_model.pth')

#---------------------- training, validation loss vs epochs ---------------
# Plot training vs validation loss
plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#-------------------------------------------------------------------------------------

#--------------------- to plot specific targets vs prediction --------------
# Convert the lists to numpy arrays
val_targets = np.array(val_targets)
val_predictions = np.array(val_predictions)


def mae(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.abs(y_true - predictions))



# Returns: 0.833
ind=61000 - 61000*9 -1

# Plot targets vs predictions for the validation set
plt.scatter(val_targets[ind:,0]*1.1078, val_predictions[ind:,0]*1.1078)
plt.xlabel('Targets')
plt.ylabel('Predictions')
plt.title('Targets vs Predictions (Validation Set) - Thermal conductivity, k')
plt.show()
print(mae(val_targets[ind:,0], val_predictions[ind:,0]))

# Plot targets vs predictions for the validation set
plt.scatter(val_targets[ind:,1]*4370000, val_predictions[ind:,1]*4370000)
plt.xlabel('Targets')
plt.ylabel('Predictions')
plt.title('Targets vs Predictions (Validation Set) - Heat capacity, rho*Cp')
plt.show()

# Plot targets vs predictions for the validation set
plt.scatter(val_targets[ind:,2]*200, val_predictions[ind:,2]*200)
plt.xlabel('Targets')
plt.ylabel('Predictions')
plt.title('Targets vs Predictions (Validation Set) - Thickness')
plt.show()
print(mae(val_targets[ind:,2], val_predictions[ind:,2]))

# Plot targets vs predictions for the validation set
plt.scatter(val_targets[ind:,3]*5.04, val_predictions[ind:,3]*5.04)
plt.xlabel('Targets')
plt.ylabel('Predictions')
plt.title('Targets vs Predictions (Validation Set) - Refractive Index')
plt.show()


# Plot targets vs predictions for the validation set
plt.scatter(val_targets[ind:,3]*5.04, val_predictions[ind:,3]*5.04, color= "blue", 
            marker= "o")
plt.rcParams['axes.facecolor'] = 'white'
plt.xlabel('Targets')
plt.ylabel('Predictions')
plt.title('Targets vs Predictions (Validation Set) - Refractive Index')
# Setting the background color of the plot
# using set_facecolor() method
# ax = plt.axes()
# ax.set_facecolor("white")
#############################
plt.show()
###################################################
# --------------------------- to validate with exp data ------------------
###############################################################################


# Load the saved fusion model
fusion_model = torch.load('fusion_model.pth')



IR = np.array(pd.read_csv('IR_Data_withlabels_3000_sim_48_exp.csv', header = None)) # simulation and exp data
# # df_IR = np.array(pd.read_csv('IR_Data_withlabels_3000_sim.csv', header = None)) # simulation data

Thz = np.array(pd.read_csv('Thz_data_withlabels_3000_sim_48_exp.csv', header = None)) # simulation and exp data
# # df_Thz = np.array(pd.read_csv('Thz_data_withlabels_3000_sim.csv', header = None)) # simulation data

X_IR_new = IR[3001:,4:]
X_Thz_new = Thz[3001:,6:206]
Y_new=Thz[1+3000:,:4] #labels - k, rhocp, thickness, refractive index

new_data1=X_IR_new #data1
new_data2=X_Thz_new #data2

# Convert the new validation data to tensors
new_data1_tensor = torch.tensor(new_data1).float()
new_data2_tensor = torch.tensor(new_data1).float()

# Set the model to evaluation mode
fusion_model.eval()
# Pass the new validation data through the fusion model
with torch.no_grad():
    outputs = fusion_model(new_data1_tensor, new_data2_tensor)
    # Compute loss
    # loss = criterion(outputs, Y_new)
    
    # Extract the predicted properties from the output
    predictions = outputs.squeeze().tolist()

cc=np.array(predictions)


p1 = cc[:,0]
p2 = cc[:,1]
p3 = cc[:,2]
p4 = cc[:,3]

#-----------------------------------------------------------------------

s1_p1_targ=Y_new[:12,0]
s1_p2_targ=Y_new[:12,1]*0.000001
s1_p3_targ=Y_new[:12,2]*1000000
s1_p4_targ=Y_new[:12,3]

s2_p1_targ=Y_new[12:24,0]
s2_p2_targ=Y_new[12:24,1]*0.000001
s2_p3_targ=Y_new[12:24,2]*1000000
s2_p4_targ=Y_new[12:24,3]

s3_p1_targ=Y_new[24:36,0]
s3_p2_targ=Y_new[24:36,1]*0.000001
s3_p3_targ=Y_new[24:36,2]*1000000
s3_p4_targ=Y_new[24:36,3]

s4_p1_targ=Y_new[36:48,0]
s4_p2_targ=Y_new[36:48,1]*0.000001
s4_p3_targ=Y_new[36:48,2]*1000000
s4_p4_targ=Y_new[36:48,3]


s1_p1_pred=cc[:12,0]*1.1078
s1_p2_pred=cc[:12,1]*4.37
s1_p3_pred=cc[:12,2]*200
s1_p4_pred=cc[:12,3]*5.04

s2_p1_pred=cc[12:24,0]*1.1078
s2_p2_pred=cc[12:24,1]*4.37
s2_p3_pred=cc[12:24,2]*200
s2_p4_pred=cc[12:24,3]*5.04

s3_p1_pred=cc[24:36,0]*1.1078
s3_p2_pred=cc[24:36,1]*4.37
s3_p3_pred=cc[24:36,2]*200
s3_p4_pred=cc[24:36,3]*5.04

s4_p1_pred=cc[36:48,0]*1.1078
s4_p2_pred=cc[36:48,1]*4.37
s4_p3_pred=cc[36:48,2]*200
s4_p4_pred=cc[36:48,3]*5.04
#-----------------------------------------------------------------------


# p1_norm = p1/np.max(p1)
# p2_norm = p2/np.max(p2)
# p3_norm = p3/np.max(p3)
# p4_norm = p4/np.max(p4)

# xlist = list(np.arange(1,48+1))
# print(xlist)
# xlistarray=np.array(xlist)


# Plot targets vs predictions for the validation set
plt.scatter(Y_new[:,2]*1000000,p3*200)
plt.xlabel('Targets')
plt.ylabel('Predictions')
plt.title('Targets vs Predictions (Validation Set) - Heat capacity, rho*Cp')
plt.show()

#------------------- experiment data - prediction plot ------------------------

# Plot targets vs predictions for the validation set
plt.scatter(Y_new[:,0], p1*1.1078)
plt.xlabel('Targets')
plt.ylabel('Predictions')
plt.title('Targets vs Predictions (Validation Set) - Thermal conductivity, k')
plt.show()
#print(mae(Y_new[:,0], p1))

# Plot targets vs predictions for the validation set
plt.scatter(Y_new[:,1], p2*4370000) # ref max value
plt.xlabel('Targets')
plt.ylabel('Predictions')
plt.title('Targets vs Predictions (Validation Set) - Heat capacity, rho*Cp')
plt.show()

# Plot targets vs predictions for the validation set
plt.scatter(Y_new[:,2]*1000000, p3*200)
plt.xlabel('Targets')
plt.ylabel('Predictions')
plt.title('Targets vs Predictions (Validation Set) - Thickness')
plt.show()
# print(mae(val_targets[53999:,2], val_predictions[53999:,2]))

# Plot targets vs predictions for the validation set
plt.scatter(Y_new[:,3], p4*(5.04))
plt.xlabel('Targets')
plt.ylabel('Predictions')
plt.title('Targets vs Predictions (Validation Set) - Refractive Index')
plt.show()
#Adding the mean of the plot 
# plt.axvline(p4["refractive index"].mean(), color='r')
#-------------------------------------------------------------------------------
#------------------------------ plot box plot of prediction -----------------------

p_exppred=p4*200 #predicted properties using exp data
p_4samples= [s1_p3_pred, s2_p3_pred ,s3_p3_pred, s4_p3_pred]

fig = plt.figure(figsize =(10, 7))
# Creating axes instance
ax = fig.add_axes([0, 0, 1, 1])
# Creating plot
bp = ax.boxplot(p_4samples)
 
# show plot
plt.show()

#----------------------------------------------------------------------------------
#-------------------------------- MAE MSE values ----------------------------------


# Convert the ground truth to a numpy array
Y_ptarget = 1.05
Y_ppred=s4_p1_pred

# Calculate Mean Squared Error (MSE)
# rmse = np.sqrt(np.mean((Y_ppred - Y_ptarget) ** 2))

# Calculate Mean Absolute Error (MAE)
MAPE = (np.mean(np.abs((Y_ptarget - Y_ppred)/Y_ptarget)))*100

print((np.mean(Y_ppred)))
print(np.std(Y_ppred))
# Print the evaluation metrics
# print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", MAPE)
