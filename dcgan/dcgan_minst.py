import torch
import torchvision
import torchvision.utils
import torch.nn
import numpy

transform=torchvision.transforms.Compose([
    torchvision.transforms.Resize(32),#将28*28 -》32*32
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5], [0.5])
])

test_data=torchvision.datasets.MNIST(
    root='./minst_data/',#路径
    transform=transform,#数据处理
    train=True,#使用测试集，这个看心情
    download=True#下载
)

test_data_load=torch.utils.data.DataLoader(
    dataset=test_data,
    shuffle=True,#每次打乱顺序
    batch_size=64#批大小，这里根据数据的样本数量而定，最好是能整除
)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Generater(torch.nn.Module):
    def __init__(self):
        super(Generater, self).__init__()

        self.lay1=torch.nn.Sequential(
            torch.nn.Linear(100, 128*1*8*8)
        )
        self.lay2 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(128),

            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(128, 128, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128, 0.8),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(128, 64, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64, 0.8),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Conv2d(64, 1, 3, stride=1, padding=1),
            torch.nn.Tanh(),
        )
    def forward(self, z):
        out = self.lay1(z)
        out = out.view(out.shape[0], 128, 8, 8)
        img = self.lay2(out)
        return img

class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.lay1=torch.nn.Sequential(
            torch.nn.Conv2d(1,16,3,2,1),
            torch.nn.LeakyReLU(0.2,inplace=True),
            torch.nn.Dropout2d(0.25),

            torch.nn.Conv2d(16, 32, 3, 2, 1),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Dropout2d(0.25),
            torch.nn.BatchNorm2d(32, 0.8),

            torch.nn.Conv2d(32, 64, 3, 2, 1),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Dropout2d(0.25),
            torch.nn.BatchNorm2d(64, 0.8),

            torch.nn.Conv2d(64, 128, 3, 2, 1),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Dropout2d(0.25),
            torch.nn.BatchNorm2d(128, 0.8),
        )

        self.lay2=torch.nn.Sequential(
            torch.nn.Linear(128 * (32 // 2 ** 4) ** 2, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, img):
        out = self.lay1(img)
        out = out.view(out.shape[0], -1)
        validity = self.lay2(out)
        return validity


# Loss function
g_net=Generater().cuda()
d_net=Discriminator().cuda()

#损失函数，优化器
loss_fun=torch.nn.BCELoss().cuda()

# Initialize weights
g_net.apply(weights_init_normal)
d_net.apply(weights_init_normal)

g_optimizer=torch.optim.Adam(g_net.parameters(),lr=0.0002,betas=(0.5,0.999))
d_optimizer=torch.optim.Adam(d_net.parameters(),lr=0.0002,betas=(0.5,0.999))
epoch_n=20

for epoch in range(epoch_n):
    for i,(img,_) in enumerate(test_data_load):

        img_num=img.size(0)
        real_lab = torch.autograd.Variable(torch.cuda.FloatTensor(img.size(0), 1).fill_(1.0), requires_grad=False)#全1
        fake_lab=torch.autograd.Variable(torch.cuda.FloatTensor(img.size(0), 1).fill_(0.0), requires_grad=False)#全0
        real_img=torch.autograd.Variable(img.type(torch.cuda.FloatTensor))

        #训练G
        g_optimizer.zero_grad()
        z=torch.autograd.Variable(torch.cuda.FloatTensor(numpy.random.normal(0,1,(img_num,100))))#噪声z满足标准正态分布

        fake_img=g_net(z)
        feke_img_lab=d_net(fake_img)

        g_loss=loss_fun(feke_img_lab,real_lab)
        g_loss.backward()
        g_optimizer.step()

        #D
        d_optimizer.zero_grad()

        real_img_lab=d_net(real_img)
        real_loss=loss_fun(real_img_lab,real_lab)

        fake_img_lab=d_net(fake_img.detach())
        fake_loss=loss_fun(fake_img_lab,fake_lab)

        d_loss=(real_loss+fake_loss)/2
        d_loss.backward()
        d_optimizer.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, epoch_n, i, len(test_data_load), d_loss.item(), g_loss.item())
        )
        batches_done = epoch * len(test_data_load) + i
        if batches_done % 400 == 0:
            torchvision.utils.save_image(fake_img.data[:25], "./result_dcgan_minst/%d.png" % batches_done, nrow=5, normalize=True)
