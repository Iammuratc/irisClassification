# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 00:39:20 2019

@author: Admin
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class IrisDataSet(Dataset):
    def __init__(self,csv_file):
        self.csv_file=csv_file
        self.df = pd.read_csv('iris.txt', sep=',')
        self.df["species"] = self.df["species"].map({
                            "setosa": 0,
                            "versicolor": 1,
                            "virginica": 2}).astype(int)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
#        X = self.df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
        
        X=np.array(self.df.iloc[idx,:4])
#        X=np.array(X)
#        print(X[0])
#        y_linear = self.df['species']
        y_linear=self.df.iloc[idx,4]
        y = np.zeros((3,))
        y[y_linear] = 1
        

        sample={'properties':X,'label':y}    
        return sample
    
class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4,100)
        self.fc2 = nn.Linear(100,100)
        self.fc3 = nn.Linear(100,3)
#        self.fc4 = nn.Linear(10,3)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
#        x = F.relu(self.fc3(x))
#        x = self.fc4(x)
#        print(x)
        yhat = F.log_softmax(x,dim=1)
#        print(yhat)
#        print(y.sum(0))
        return yhat
    
class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss

my_data=IrisDataSet('iris.txt')


#for i,x in enumerate(dataloader):
#    print(x['properties'])
#    break
train_size = int(0.8 * len(my_data))
test_size = len(my_data) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(my_data, [train_size, test_size])

train_data = DataLoader(train_dataset,batch_size=10,shuffle=True)
test_data= DataLoader(test_dataset,batch_size=10)

    
#print(X[1,])
my_net = NN()

#for data in train_data:
#    X,y=data['properties'],data['label']
#    X=X.float()
#    y=y.float()
#    output=my_net(X)
#    print(output)
#    break    
    
optimizer=optim.Adam(my_net.parameters(),lr=0.0001)

epochs=100

for epoch in range(epochs):
    for data in train_data:
        X,y=data['properties'],data['label']
        X=X.float()
        y=y.float()
#        print(X.shape,y.shape)
        my_net.zero_grad()
        output=my_net(X)
#        print(output)
#        break
#        print(X.shape,output.shape,y.shape)
        rmse=RMSELoss()
        loss=rmse(output,y)
        loss.backward()
        optimizer.step()
    print(loss)
                

correct=0
total=0

with torch.no_grad():
    for data in train_data:
        X,y=data['properties'],data['label']
        X=X.float()
        y=y.float()
        output=my_net(X)
#        print(output)
#        print(y.shape,output.shape)
#        break
        for idx, i in enumerate(output):
            
            if torch.argmax(i) == torch.argmax(y[idx]):
                correct += 1
            total += 1
print('Accuracy: {}'.format(round(correct/total,3)))
        