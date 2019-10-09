import torch
import torchvision
import torch.utils.data
import torch.nn
import numpy
import torch.autograd
import torchvision.utils
import torch.optim

#图像读入与处理
transform=torchvision.transforms.Compose([
    torchvision.transforms.Resize(28,28),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5], [0.5])
])

#数据集
test_data=torchvision.datasets.MNIST(
    root='./minst_data/',#路径
    transform=transform,#数据处理
    train=True,#使用测试集，这个看心情
    download=True#下载
)

#数据加载器， DataLoader就是用来包装所使用的数据，每次抛出一批数据
test_data_load=torch.utils.data.DataLoader(
    dataset=test_data,
    shuffle=True,#每次打乱顺序
    batch_size=64#批大小，这里根据数据的样本数量而定，最好是能整除
)

class Generater(torch.nn.Module):
    def __init__(self):
        super(Generater, self).__init__()
        self.lay = torch.nn.Sequential(
            torch.nn.Linear(100, 128),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Linear(128, 256),
            torch.nn.BatchNorm1d(256, 0.8),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Linear(256, 512),
            torch.nn.BatchNorm1d(512, 0.8),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Linear(512, 1024),
            torch.nn.BatchNorm1d(1024, 0.8),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Linear(1024, 784),
            torch.nn.Tanh()
        )

    def forward(self, z):
        x = self.lay(z)
        img = x.view(x.size(0), 1,28,28)
        return img

class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.D_lay1=torch.nn.Sequential(

            torch.nn.Linear(784,512),
            torch.nn.LeakyReLU(0.2,inplace=True),

            torch.nn.Linear(512, 256),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Linear(256,1),
        )
    def forward(self, img):
        z=img.view(img.size(0),-1)#28*28*1的图转化成784向量
        lab=self.D_lay1(z)
        return lab


g_net=Generater().cuda()
d_net=Discriminator().cuda()

optimizer_g=torch.optim.RMSprop(g_net.parameters(), lr=0.00005)
optimizer_d=torch.optim.RMSprop(d_net.parameters(),lr=0.00005)
epoch_n=20

for epoch in range(epoch_n):
    for i,(img,_) in enumerate(test_data_load):

        img_num = img.size(0)
        real_img = torch.autograd.Variable(img.type(torch.cuda.FloatTensor))

        #D
        optimizer_d.zero_grad()
        z = torch.autograd.Variable(torch.cuda.FloatTensor(numpy.random.normal(0, 1, (img_num, 100))))  # 噪声z满足标准正态分布

        fake_img=g_net(z)

        d_fake_img_loss=d_net(fake_img.detach())
        d_real_img_loss=d_net(real_img)
        d_loss=-torch.mean(d_real_img_loss)+torch.mean(d_fake_img_loss)

        d_loss.backward()
        optimizer_d.step()

        for p in d_net.parameters():
            p.data.clamp_(-0.01,0.01)

        if i%5==0:
            #G
            optimizer_g.zero_grad()
            fake_img=g_net(z)
            g_loss=-torch.mean(d_net(fake_img))

            g_loss.backward()
            optimizer_g.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, epoch_n, i, len(test_data_load), d_loss.item(), g_loss.item())
            )

        batches_done = epoch * len(test_data_load) + i
        if batches_done % 400 == 0:
            torchvision.utils.save_image(fake_img.data[:25], "./result_wgan_minst/%d.png" % batches_done, nrow=5,
                                             normalize=True)





