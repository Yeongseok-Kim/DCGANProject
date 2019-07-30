import torch
from torch.utils.data.dataset import Dataset

from PIL import Image

import os

import torchvision
import torchvision.transforms as transforms



class MakeDataset(Dataset):
    def __init__(self,img_dir):
        self.img_list=[img_dir+file_name for file_name in os.listdir(img_dir)]
        self.transform=transforms.Compose([transforms.Resize((64,64)),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self,index):
        img=Image.open(self.img_list[index])
        img=self.transform(img)
        label=torch.ones(1)
        fake_label=torch.zeros(1)
        return img,label,fake_label