#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 00:41:46 2019
LeNet 5 Pytorch
@author: wuzhenglung
"""

import torch as t
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms

import matplotlib
import matplotlib.pyplot as plt

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        
        self.model=nn.Sequential(
                nn.Conv2d(1,6,5,padding=2),
                nn.ReLU(),
                nn.AvgPool2d(2,stride=2),
                #nn.ReLU(),
                nn.Conv2d(6,16,5),
                nn.ReLU(),
                nn.AvgPool2d(2,stride=2),
                #nn.ReLU(),
                nn.Conv2d(16,120,5),
                nn.ReLU(),                
                )
        self.model2=nn.Sequential(
                nn.Linear(120,84),
                nn.ReLU(),
                nn.Linear(84,10),
                nn.ReLU(),
                nn.Softmax(dim=-1)               
                )
    
    def forward(self,img):
        
        x=self.model(img)
        x=x.view(x.size(0),-1)
        x=self.model2(x)
        return x
                   

epoch=10
batch_size=1000
rootpath='/Users/wuzhenglung/Desktop/pytorch test'


#dataset import 

data_transform=transforms.Compose([transforms.ToTensor()])

train_dataset=torchvision.datasets.MNIST(
        rootpath,train=True,download=True,transform=data_transform)

train_loader=t.utils.data.DataLoader(
        train_dataset,batch_size=batch_size,shuffle=True)

verify_dataset=torchvision.datasets.MNIST(
        rootpath,train=False,download=True,transform=data_transform)

verify_loader=t.utils.data.DataLoader(
        verify_dataset,batch_size=batch_size,shuffle=True)




Net=LeNet()
loss_func=nn.CrossEntropyLoss()
optimizer=optim.SGD(Net.parameters(),lr=2e-2)

accur=[]
for j in range(epoch):
    print('start epoch',j+1)
    print('training start:')
    
    
    for i,(data,label) in enumerate(train_loader):
        optimizer.zero_grad()
        output=Net(data)    
        loss=loss_func(output,label)
        loss.backward()
        optimizer.step()
        
       
    print('start verifying:')
    total_corr=0
    
    for i,(data,label) in enumerate(verify_loader):
        output=Net(data)
        _,pred=output.max(1)        
        total_corr+=pred.eq(label.view_as(pred)).sum()
        print(float(total_corr))
        print('accuracy:%f ' %(float(total_corr)/((i+1)*batch_size)))
    accur.append(float(total_corr)*100/(len(verify_loader)*batch_size))

# Data for plotting


fig, ax = plt.subplots()
ax.plot(accur)

ax.set(xlabel='epoch', ylabel='accuracy (%)',
       title='ReLU,Softmax')
ax.grid()

#fig.savefig("test.png")
plt.show()
        
        
        
    
   



