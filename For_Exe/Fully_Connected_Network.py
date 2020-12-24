#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torchvision import transforms, datasets, models
import torch.nn as nn


# In[5]:


def double_conv2D(in_channel,out_channel):
    conv= nn.Sequential(
        nn.Conv2d(in_channel,out_channel,kernel_size=3,padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channel,out_channel,kernel_size=3,padding=1),
        nn.ReLU(inplace=True))
    return conv


# In[6]:


def tripple_conv2D(in_channel,out_channel):
    conv= nn.Sequential(
        nn.Conv2d(in_channel,out_channel,kernel_size=3,padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channel,out_channel,kernel_size=3,padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channel,out_channel,kernel_size=3,padding=1),
        nn.ReLU(inplace=True))
    return conv


# In[33]:


class FC(nn.Module):
    def __init__(self):
        super(FC,self).__init__()
        self.conv1= double_conv2D(1,64)
        self.maxpool2D_1 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.conv2= double_conv2D(64,128)
        self.maxpool2D_2 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.conv3= tripple_conv2D(128,256)
        self.maxpool2D_3 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.conv4= tripple_conv2D(256,512)
        self.maxpool2D_4 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.conv5= tripple_conv2D(512,512)
        self.maxpool2D_5 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.conv6= nn.Sequential(nn.Conv2d(512,4096,kernel_size=3,padding=1),nn.ReLU(inplace=True))
        self.conv7= nn.Sequential(nn.Conv2d(4096,4096,kernel_size=1,padding=0),nn.ReLU(inplace=True))
        
        self.score5=nn.Sequential(nn.Conv2d(4096,1,kernel_size=1,padding=0),nn.ReLU(inplace=True))
        self.score4=nn.Sequential(nn.Conv2d(512,1,kernel_size=1,padding=0),nn.ReLU(inplace=True))
        self.score3=nn.Sequential(nn.Conv2d(256,1,kernel_size=1,padding=0),nn.ReLU(inplace=True))
        
        self.upscore1 = nn.ConvTranspose2d(1,1,kernel_size=4,stride=2,padding=1)
        self.upscore2 = nn.ConvTranspose2d(1,1,kernel_size=4,stride=2,padding=1)
        self.upscore3 = nn.ConvTranspose2d(1,1,kernel_size=16,stride=8,padding=4)
        
    def forward(self,img):
        #encorder
        x1 = self.conv1(img)

        x2 = self.maxpool2D_1(x1)

        x3 = self.conv2(x2)
        x4 = self.maxpool2D_2(x3)
        x5 = self.conv3(x4)
        x6 = self.maxpool2D_3(x5)
        x7 = self.conv4(x6)
        x8 = self.maxpool2D_4(x7)
        x9 = self.conv5(x8)
        x10 = self.maxpool2D_5(x9)
        x11 = self.conv6(x10)

        x12 = self.conv7(x11)
        
        s5 = self.score5(x12)
        s4 = self.score4(x8)
        s3 = self.score3(x6)
        
        us1= self.upscore1(s5)
        summ1 = us1+s4
        us2 = self.upscore2(summ1)
        summ2= us2+s3
        us3 = self.upscore3(summ2)
        return us3

#for testing
"""img = torch.rand((4,1,224,224))
model = FC()
model(img)"""