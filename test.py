import torch

from matplotlib import pyplot as plt

import model

if __name__=='__main__':
    net=model.Generator()

    # 네트워크 데이터 로드
    # model_epoch_1.pth 수정해주세요
    load_net='./model_epoch_1.pth'
    net.load_state_dict(torch.load(load_net))

    with torch.no_grad():
        noise1=torch.randn(1,100,1,1)
        fake_img1=net(noise1)
        noise2=torch.randn(1,100,1,1)
        fake_img2=net(noise2)
        noise3=torch.randn(1,100,1,1)
        fake_img3=net(noise3)
        noise4=torch.randn(1,100,1,1)
        fake_img4=net(noise4)
        noise5=torch.randn(1,100,1,1)
        fake_img5=net(noise5)
                
        fig=plt.figure()
        ax1=fig.add_subplot(1,5,1)
        ax2=fig.add_subplot(1,5,2)
        ax3=fig.add_subplot(1,5,3)
        ax4=fig.add_subplot(1,5,4)
        ax5=fig.add_subplot(1,5,5)

        ax1.imshow(fake_img1.squeeze().permute(1,2,0).numpy().clip(0,1))
        ax2.imshow(fake_img2.squeeze().permute(1,2,0).numpy().clip(0,1))
        ax3.imshow(fake_img3.squeeze().permute(1,2,0).numpy().clip(0,1))
        ax4.imshow(fake_img4.squeeze().permute(1,2,0).numpy().clip(0,1))
        ax5.imshow(fake_img5.squeeze().permute(1,2,0).numpy().clip(0,1))

        plt.show()