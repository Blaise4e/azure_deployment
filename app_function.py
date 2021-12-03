import numpy as np
import os
import time
import matplotlib.pyplot as mp
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as om
import torchvision as tv
import torch.utils.data as dat
from PIL import Image, ImageOps

class MedNet(nn.Module):
    def __init__(self,xDim,yDim,numC):    
        super(MedNet,self).__init__()         
        numConvs1 = 5
        convSize1 = 7
        numConvs2 = 10
        convSize2 = 7
        numNodesToFC = numConvs2*(xDim-(convSize1-1)-(convSize2-1))*(yDim-(convSize1-1)-(convSize2-1))

        self.cnv1 = nn.Conv2d(1, numConvs1, convSize1)
        self.cnv2 = nn.Conv2d(numConvs1, numConvs2, convSize2)
        
        fcSize1 = 400
        fcSize2 = 80
                
        self.ful1 = nn.Linear(numNodesToFC,fcSize1)
        self.ful2 = nn.Linear(fcSize1, fcSize2)
        self.ful3 = nn.Linear(fcSize2,numC)
        
    def forward(self,x):
        
        x = F.elu(self.cnv1(x)) # Feed through first convolutional layer, then apply activation
        x = F.elu(self.cnv2(x)) # Feed through second convolutional layer, apply activation
        x = x.view(-1,self.num_flat_features(x)) # Flatten convolutional layer into fully connected layer
        x = F.elu(self.ful1(x)) # Feed through first fully connected layer, apply activation
        x = F.elu(self.ful2(x)) # Feed through second FC layer, apply output
        x = self.ful3(x)        # Final FC layer to output. No activation, because it's used to calculate loss
        return x

    def num_flat_features(self, x):  # Count the individual nodes in a layer
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def scaleImage(y):          # Pass a PIL image, return a tensor

    if(y.min() < y.max()):  # Assuming the image isn't empty, rescale so its values run from 0 to 1
        y = (y - y.min())/(y.max() - y.min()) 
    z = y - y.mean()        # Subtract the mean value of the image
    return z

    
def predict_img(img, model):
    img = Image.open(img)
    #Transform to gray (1 channel)
    img = ImageOps.grayscale(img)
    #Resize the img
    size = 64, 64
    img = img.resize(size)
    #Convert to a matrix 
    toTensor = tv.transforms.ToTensor()
    img = toTensor(img)
    img.shape
    #Reshape for the model
    img = img.reshape([1,1,64,64])
    img=scaleImage(img) 
    classname = ['AbdomenCT', 'BreastMRI', 'ChestCT', 'CXR', 'Hand', 'HeadCT']
    yOut = model(img)
    max=yOut.max(1)[1].tolist()[0]
    pred=classname[max]
    return pred