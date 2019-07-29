import torch
import torch.nn as nn



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.layer=nn.Sequential(
        nn.Conv2d(3,64,4,2,1),
        nn.LeakyReLU(0.2),

        nn.Conv2d(64,128,4,2,1),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2),

        nn.Conv2d(128,256,4,2,1),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2),

        nn.Conv2d(256,512,4,2,1),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.2),

        nn.Conv2d(512,1024,4,2,1),
        nn.BatchNorm2d(1024),
        nn.LeakyReLU(0.2),

        nn.Conv2d(1024,1,2,1,0),
        nn.Sigmoid())
        
    def forward(self,x):
        out=self.layer(x)
        return out



class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.layer=nn.Sequential(
        nn.ConvTranspose2d(100,1024,4,1,0),
        nn.BatchNorm2d(1024),
        nn.ReLU(),
        
        nn.ConvTranspose2d(1024,512,4,2,1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        
        nn.ConvTranspose2d(512,256,4,2,1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        
        nn.ConvTranspose2d(256,128,4,2,1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        
        nn.ConvTranspose2d(128,3,4,2,1),
        nn.Tanh())
        
    def forward(self,x):
        out=self.layer(x)
        return out