import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import dataset

import model

if __name__=='__main__':
    batch_size=64
    
    # 디바이스 식별
    device='cuda' if torch.cuda.is_available() else 'cpu'
    
    # 트레이닝 데이터 로드
    # './data/Hands/' 데이터 경로로 수정해주세요
    train_data=dataset.MakeDataset('./data/Hands/')
    train_set=DataLoader(train_data,batch_size,True)
    
    # 트레이닝
    # training_epochs (학습 원하는 만큼) 수정해주세요
    learning_rate=0.0002
    training_epochs=80
    
    d_net=model.Discriminator().to(device)
    g_net=model.Generator().to(device)
    
    criterion=nn.BCELoss()
    
    d_optimizer=optim.Adam(d_net.parameters(),learning_rate)
    g_optimizer=optim.Adam(g_net.parameters(),learning_rate)
    
    print('Learning started. It takes sometime.')
    
    for epoch in range(training_epochs):
        for img,label,fake_label in train_set:
            img=img.to(device)
            label=label.to(device)
            fake_label=fake_label.to(device)
            
            batch_size=img.size(0)
            
            d_optimizer.zero_grad()
            g_optimizer.zero_grad()
            
            hypothesis=d_net(img)
            d_real_cost=criterion(hypothesis,label)
            
            noise=torch.randn(batch_size,100,1,1).to(device)
            fake_img=g_net(noise)
            hypothesis=d_net(fake_img)
            d_fake_cost=criterion(hypothesis,fake_label)
            
            d_cost=d_real_cost+d_fake_cost
            d_cost.backward(retain_graph=True)
            d_optimizer.step()
            
            g_cost=criterion(hypothesis,label)
            g_cost.backward()
            g_optimizer.step()
            
            print('D cost:',d_cost.item())
            print('G cost:',g_cost.item())
            
        print('epoch',epoch+1,'is over.')
        
        torch.save(g_net.state_dict(),'color_model_epoch_%d.pth'%(epoch+1))
        
    print('Learning Finished!')