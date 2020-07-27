"""
authors = Sourabh Hanamsheth, Ruta Kulkarni
Code to train the neural network with the skeleton dataset

"""

import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import glob
import getData
import os
import pcd_operations

pcd = pcd_operations.PCD()
dataloader = getData.DataLoader()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Classifier(nn.Module):
    def __init__(self, ip, H1, H2, H3, H4, H5, H6, op):
        super().__init__()
        self.linear1 = nn.Linear(ip, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, H3)
        self.linear4 = nn.Linear(H3, H4)
        self.linear5 = nn.Linear(H4, H5)
        self.linear6 = nn.Linear(H5, H6)
        self.linear7 = nn.Linear(H6, op)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = F.relu(self.linear5(x))
        x = F.relu(self.linear6(x))
        return self.linear7(x)
    
model = Classifier(1800, 1224, 812, 550, 206, 128, 64, 18).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.00001,weight_decay=0.001)

epochs = 100
running_loss_history = []
running_corrects_history = []
running_val_loss_history = []
running_val_corrects_history = []

# for the number of epochs

for run in range(epochs):
    running_loss = 0.0
    corrects = 0.0
    running_val_loss = 0.0
    val_corrects = 0.0
    count=0
    for file in glob.glob("./Data/train/*npy"):
        count+=1
        # print(count)
        skeleton = np.load(file)
        # pcd.viz([skeleton])
     
        skeleton = dataloader.normalize(skeleton)
        # pcd.viz([skeleton])

        skeleton = (dataloader.preprocess(skeleton)).ravel()
        skeleton = torch.tensor(skeleton).view(1,-1).to(device)
        
        filename = os.path.basename(file)
        label = np.array([(int(filename[1:3])-1)]) 

        label = torch.tensor(label).to(device)
        # labels = dataloader.oneHotEncoding(label)
        # labels= torch.Tensor(labels).to(device)
        

        output = model.forward(skeleton.float())
        _, pred = torch.max(output, 1)

        corrects += torch.sum(pred == label)
        loss = criterion(output, label)
        running_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    else:
        epoch_corrects = corrects/500
        epoch_loss = running_loss/500
        
        
        for file in glob.glob("./Data/test/*npy"):
            
            val_skeleton = np.load(file)
            
            val_skeleton = dataloader.normalize(val_skeleton)
            val_skeleton = (dataloader.preprocess(val_skeleton)).ravel()
            val_skeleton = torch.tensor(val_skeleton).view(1,-1).to(device)
            
            filename = os.path.basename(file)
            val_label = np.array([(int(filename[1:3])-1)]) 
    
            val_label = torch.tensor(val_label).to(device)
            
            val_output = model.forward(val_skeleton.float())
            _,val_pred = torch.max(val_output, 1)            
            
            

            val_corrects += torch.sum(val_pred == val_label)
            val_loss = criterion(val_output, val_label)
            running_val_loss += val_loss
            
            
        epoch_val_corrects = val_corrects/40
        epoch_val_loss = running_val_loss/40
            
        
    running_loss_history.append(epoch_loss)
    running_corrects_history.append(epoch_corrects)
        
    running_val_loss_history.append(epoch_val_loss)
    running_val_corrects_history.append(epoch_val_corrects)
    
    print('Epochs : {}'.format(run))
    print("traing loss :{:.4f}, acc :{:.4f}".format(epoch_loss, epoch_corrects))
    print("validation loss :{:.4f}, acc :{:.4f}".format(epoch_val_loss, epoch_val_corrects))


# plot the loss
ax = plt.figure()
plt.plot(range(epochs), running_loss_history, label="training loss")
plt.plot(running_val_loss_history, label = "validation loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Kinect - Activity Recognition')
plt.legend()
plt.grid()
plt.show()


# plot the accuracy
plt.figure()
plt.plot(range(epochs), running_corrects_history, label="training acc")
plt.plot(running_val_corrects_history, label = "validation acc")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Kinect - Activity Recognition')
plt.legend()
plt.grid()
plt.show()

torch.save(model.state_dict(), "./kinect_activity_4.pth")