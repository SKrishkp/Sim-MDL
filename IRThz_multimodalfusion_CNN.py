
"""
@author: krishna
"""
#################################################################################################
# multi-modal fusion - CNN - IR and Thz signals
#################################################################################################

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
noise1 = np.random.normal(loc=0, scale=0.05, size=data1_nonoise.shape)  # Adjust scale as desired
# Add white noise to the input data
noisy_data1 = data1_nonoise + noise1

# Generate white noise
noise2 = np.random.normal(loc=0, scale=0.05, size=data2_nonoise.shape)  # Adjust scale as desired
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

# Define CNN model 1
class CNNModel1(nn.Module):
    def __init__(self):
        super(CNNModel1, self).__init__()
        # Define the layers of CNN model 1
        # Define the layers and operations for CNN model 1
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=100, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        # ...
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        # ...
        
        # Forward pass of CNN model 1
        return x

# Define CNN model 2
class CNNModel2(nn.Module):
    def __init__(self):
        super(CNNModel2, self).__init__()
        # Define the layers of CNN model 2
        # Define the layers and operations for CNN model 2
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=100, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
    
    def forward(self, x):
        # Forward pass of CNN model 2
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        
        return x

# Define the fusion model
class FusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FusionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x1, x2):
        # Concatenate the outputs from CNN model 1 and model 2
        fused_representation = torch.cat((x1, x2), dim=1)

        # Apply fusion layers
        x = self.fc1(fused_representation)
        x = torch.relu(x)
        x = self.fc2(x)

        return x
    
###############################################################################
# hidden_size1=16
# hidden_size2=16
# fusion_size=32
# num_classes=4

# class FusionModel(nn.Module):
#     def __init__(self):
#         super(FusionModel, self).__init__()

#         # CNN model 1
#         self.cnn_model1 = CNNModel1()

#         # CNN model 2
#         self.cnn_model2 = CNNModel2()

#         # Fusion operation
#         self.fusion_layer = nn.Linear(hidden_size1 + hidden_size2, fusion_size)

#         # Activation function
#         self.activation = nn.ReLU()

#         # Output layer
#         self.output_layer = nn.Linear(fusion_size, num_classes)

#     def forward(self, input1, input2):
#         # Process input1 through CNN model 1
#         output1 = self.cnn_model1(input1)

#         # Process input2 through CNN model 2
#         output2 = self.cnn_model2(input2)

#         # Concatenate the outputs from both CNN models
#         fusion_output = torch.cat((output1, output2), dim=1)

#         # Apply activation function
#         fusion_output = self.activation(fusion_output)

#         # Perform fusion operation
#         fusion_output = self.fusion_layer(fusion_output)

#         # Apply activation function
#         fusion_output = self.activation(fusion_output)

#         # Predict the output
#         output = self.output_layer(fusion_output)

#         return output

# #############################################################################

    

class PropertyDataset(Dataset):
    def __init__(self, data11, data22, targets):
        self.data11 = data11
        self.data22 = data22
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        x1 = self.data11[index]
        x2 = self.data22[index]
        y = self.targets[index]
        return x1, x2, y

# Example usage
# Assuming you have prepared your data as 'data1', 'data2', and 'targets' arrays

# Convert data and targets to float type
data1x=np.array(data1)
data11 = data1x.astype(float)
data2x=np.array(data1)
data22 = data2x.astype(float)
targets=np.array(targets)
targets = targets.astype(float)


# Split into training and validation sets
train_size = int(0.8 * len(data11))
train_data1 = data11[:train_size]
train_data2 = data22[:train_size]
train_targets = targets[:train_size]
val_data1 = data11[train_size:]
val_data2 = data22[train_size:]
val_targets = targets[train_size:]


# Create dataset and dataloaders
train_dataset = PropertyDataset(train_data1, train_data2, train_targets)
val_dataset = PropertyDataset(val_data1, val_data2, val_targets)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True) # 200 works, 1st property
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False) #200works

# Model initialization
input_dim = data11.shape[1] + data22.shape[1]  # Concatenated input dimension
hidden_dim = 100 #200works
output_dim = 4  # Number of properties to predict #1works
cnn_model1 = CNNModel1()
cnn_model2 = CNNModel2()
fusion_model = FusionModel(input_dim, hidden_dim, output_dim)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(fusion_model.parameters(), lr=0.01) #0.1

# Training loop
num_epochs =50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model1.to(device)
cnn_model2.to(device)
fusion_model.to(device)

#------------------------------------------------------------------------------------

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

    for data11, data22, targets in train_dataloader:
        data11 = data11.to(device).float()  # Convert to float type
        data22 = data22.to(device).float()  # Convert to float type
        targets = targets.to(device).float()  # Convert to float type

        optimizer.zero_grad()

        # Forward pass through CNN model 1
        output1 = cnn_model1(data11)

        # Forward pass through CNN model 2
        output2 = cnn_model2(data22)

        # Forward pass through fusion model
        outputs = fusion_model(output1, output2)
        predictions = outputs.squeeze()  # Assuming the predictions are in the shape (batch_size, )
        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Store targets and predicted values
        all_targets.extend(targets.numpy())
        all_predictions.extend(predictions.detach().numpy())
        train_loss += loss.item() * data11.size(0)

    # Compute average training loss for the epoch
    train_loss /= len(train_dataset)
    
    train_losses.append(train_loss)

    # Validation loop
    fusion_model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for data11, data22, targets in val_dataloader:
            data11 = data11.to(device).float()  # Convert to float type
            data22 = data22.to(device).float()  # Convert to float type
            targets = targets.to(device).float()  # Convert to float type

            # Forward pass through CNN model 1
            output1 = cnn_model1(data11)

            # Forward pass through CNN model 2
            output2 = cnn_model2(data22)

            # Forward pass through fusion model
            outputs = fusion_model(output1, output2)

            # Compute loss
            loss = criterion(outputs, targets)

            val_loss += loss.item() * data11.size(0)
            
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

#-------------------- to plot all tragets vs prediction ---------------------
# pred_array=np.array(all_predictions)
# target_array=np.array(all_targets)

# # Plot targets vs predictions
# plt.scatter(target_array[:,3], pred_array[:,3])
# plt.xlabel('Targets')
# plt.ylabel('Predictions')
# plt.title('Targets vs Predictions')
# plt.show()


#--------------------- to plot specific targets vs prediction --------------
# Convert the lists to numpy arrays
val_targets = np.array(val_targets)
val_predictions = np.array(val_predictions)


def mae(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.abs(y_true - predictions))



# Returns: 0.833

# Plot targets vs predictions for the validation set
plt.scatter(val_targets[53999:,0]*1.1078, val_predictions[53999:,0]*1.1078)
plt.xlabel('Targets')
plt.ylabel('Predictions')
plt.title('Targets vs Predictions (Validation Set) - Thermal conductivity, k')
plt.show()
print(mae(val_targets[53999:,0], val_predictions[53999:,0]))

# Plot targets vs predictions for the validation set
plt.scatter(val_targets[53999:,1]*4370000, val_predictions[53999:,1]*4370000)
plt.xlabel('Targets')
plt.ylabel('Predictions')
plt.title('Targets vs Predictions (Validation Set) - Heat capacity, rho*Cp')
plt.show()

# Plot targets vs predictions for the validation set
plt.scatter(val_targets[53999:,2]*200, val_predictions[53999:,2]*200)
plt.xlabel('Targets')
plt.ylabel('Predictions')
plt.title('Targets vs Predictions (Validation Set) - Thickness')
plt.show()
print(mae(val_targets[53999:,2], val_predictions[53999:,2]))

# Plot targets vs predictions for the validation set
plt.scatter(val_targets[53999:,3]*5.04, val_predictions[53999:,3]*5.04)
plt.xlabel('Targets')
plt.ylabel('Predictions')
plt.title('Targets vs Predictions (Validation Set) - Refractive Index')
plt.show()


#--------------------------------------------------------------------------


# plt.savefig(weights_save_path_ + '/train_loss_vs_epochs.png')

# fig1.savefig(weights_save_path_ + '/train_loss_vs_epochs.png', dpi=200)

################################################################################
# --------------------------- to validate with exp data ------------------
###############################################################################


# # Load the saved fusion model
# fusion_model = torch.load('fusion_model.pth')



# IR = np.array(pd.read_csv('IR_Data_withlabels_3000_sim_48_exp.csv', header = None)) # simulation and exp data
# # # df_IR = np.array(pd.read_csv('IR_Data_withlabels_3000_sim.csv', header = None)) # simulation data

# Thz = np.array(pd.read_csv('Thz_data_withlabels_3000_sim_48_exp.csv', header = None)) # simulation and exp data
# # # df_Thz = np.array(pd.read_csv('Thz_data_withlabels_3000_sim.csv', header = None)) # simulation data

# X_IR_new = IR[3001:,4:]
# X_Thz_new = Thz[3001:,6:206]
# Y_new=Thz[1+3000:,:4] #labels - k, rhocp, thickness, refractive index

# new_data1=X_IR_new #data1
# new_data2=X_Thz_new #data2

# # Convert the new validation data to tensors
# new_data1_tensor = torch.tensor(new_data1).float()
# new_data2_tensor = torch.tensor(new_data1).float()

# # Set the model to evaluation mode
# fusion_model.eval()
# # Pass the new validation data through the fusion model
# with torch.no_grad():
#     outputs = fusion_model(new_data1_tensor, new_data2_tensor)
#     # Compute loss
#     # loss = criterion(outputs, Y_new)
    
#     # Extract the predicted properties from the output
#     predictions = outputs.squeeze().tolist()

# cc=np.array(predictions)


# p1 = cc[:,0]
# p2 = cc[:,1]
# p3 = cc[:,2]
# p4 = cc[:,3]

# #-----------------------------------------------------------------------

# s1_p1_targ=Y_new[:12,0]
# s1_p2_targ=Y_new[:12,1]*0.000001
# s1_p3_targ=Y_new[:12,2]*1000000
# s1_p4_targ=Y_new[:12,3]

# s2_p1_targ=Y_new[12:24,0]
# s2_p2_targ=Y_new[12:24,1]*0.000001
# s2_p3_targ=Y_new[12:24,2]*1000000
# s2_p4_targ=Y_new[12:24,3]

# s3_p1_targ=Y_new[24:36,0]
# s3_p2_targ=Y_new[24:36,1]*0.000001
# s3_p3_targ=Y_new[24:36,2]*1000000
# s3_p4_targ=Y_new[24:36,3]

# s4_p1_targ=Y_new[36:48,0]
# s4_p2_targ=Y_new[36:48,1]*0.000001
# s4_p3_targ=Y_new[36:48,2]*1000000
# s4_p4_targ=Y_new[36:48,3]


# s1_p1_pred=cc[:12,0]*1.1078
# s1_p2_pred=cc[:12,1]*4.37
# s1_p3_pred=cc[:12,2]*200
# s1_p4_pred=cc[:12,3]*5.04

# s2_p1_pred=cc[12:24,0]*1.1078
# s2_p2_pred=cc[12:24,1]*4.37
# s2_p3_pred=cc[12:24,2]*200
# s2_p4_pred=cc[12:24,3]*5.04

# s3_p1_pred=cc[24:36,0]*1.1078
# s3_p2_pred=cc[24:36,1]*4.37
# s3_p3_pred=cc[24:36,2]*200
# s3_p4_pred=cc[24:36,3]*5.04

# s4_p1_pred=cc[36:48,0]*1.1078
# s4_p2_pred=cc[36:48,1]*4.37
# s4_p3_pred=cc[36:48,2]*200
# s4_p4_pred=cc[36:48,3]*5.04
# #-----------------------------------------------------------------------


# # p1_norm = p1/np.max(p1)
# # p2_norm = p2/np.max(p2)
# # p3_norm = p3/np.max(p3)
# # p4_norm = p4/np.max(p4)

# # xlist = list(np.arange(1,48+1))
# # print(xlist)
# # xlistarray=np.array(xlist)


# # # Plot targets vs predictions for the validation set
# # plt.scatter(Y_new[:,2]*200,p3*200)
# # plt.xlabel('Targets')
# # plt.ylabel('Predictions')
# # plt.title('Targets vs Predictions (Validation Set) - Heat capacity, rho*Cp')
# # plt.show()

# #------------------- experiment data - prediction plot ------------------------

# # Plot targets vs predictions for the validation set
# plt.scatter(Y_new[:,0], p1*1.1078)
# plt.xlabel('Targets')
# plt.ylabel('Predictions')
# plt.title('Targets vs Predictions (Validation Set) - Thermal conductivity, k')
# plt.show()
# #print(mae(Y_new[:,0], p1))

# # Plot targets vs predictions for the validation set
# plt.scatter(Y_new[:,1], p2*4370000) # ref max value
# plt.xlabel('Targets')
# plt.ylabel('Predictions')
# plt.title('Targets vs Predictions (Validation Set) - Heat capacity, rho*Cp')
# plt.show()

# # Plot targets vs predictions for the validation set
# plt.scatter(Y_new[:,2]*1000000, p3*200)
# plt.xlabel('Targets')
# plt.ylabel('Predictions')
# plt.title('Targets vs Predictions (Validation Set) - Thickness')
# plt.show()
# # print(mae(val_targets[53999:,2], val_predictions[53999:,2]))

# # Plot targets vs predictions for the validation set
# plt.scatter(Y_new[:,3], p4*(5.04))
# plt.xlabel('Targets')
# plt.ylabel('Predictions')
# plt.title('Targets vs Predictions (Validation Set) - Refractive Index')
# plt.show()
# #-------------------------------------------------------------------------------
# #------------------------------ plot box plot of prediction -----------------------

# p_exppred=p4*5.04 #predicted properties using exp data
# p_4samples= [s1_p4_pred, s2_p4_pred ,s3_p4_pred, s4_p4_pred]

# fig = plt.figure(figsize =(10, 7))
# # Creating axes instance
# ax = fig.add_axes([0, 0, 1, 1])
# # Creating plot
# bp = ax.boxplot(p_4samples)
 
# # show plot
# plt.show()

# #----------------------------------------------------------------------------------
# #-------------------------------- MAE MSE values ----------------------------------


# # Convert the ground truth to a numpy array
# Y_ptarget = s1_p4_targ
# Y_ppred=s1_p4_pred

# # Calculate Mean Squared Error (MSE)
# # rmse = np.sqrt(np.mean((Y_ppred - Y_ptarget) ** 2))

# # Calculate Mean Absolute Error (MAE)
# MAPE = np.mean(np.abs((Y_ptarget - Y_ppred)/Y_ptarget))*100

# print((np.mean(Y_ppred)))
# print(np.std(Y_ppred))
# # Print the evaluation metrics
# # print("Root Mean Squared Error (RMSE):", rmse)
# print("Mean Absolute Error (MAE):", MAPE)






