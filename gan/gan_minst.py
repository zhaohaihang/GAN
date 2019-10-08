import torch
import torchvision
import torch.utils.data
import torch.nn
import numpy
import torch.autograd
import torchvision.utils

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

#将1*784向量，转换成28*28图片
def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)
    out = out.view(-1, 1, 28, 28)
    return out

#生成器
class Generater(torch.nn.Module):
    def __init__(self):
        super(Generater, self).__init__()
        self.G_lay1=torch.nn.Sequential(

            #batchnorm用于加速收敛和减缓过拟合,解决多层神经网络中间层的协方差偏移,
            # 类似于网络输入进行零均值化和方差归一化的操作，不过是在中间层的输入中操作而已
            torch.nn.Linear(100,128),
            #torch.nn.BatchNorm1d(128, 0.8),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Linear(128,256),
            torch.nn.BatchNorm1d(256, 0.8),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Linear(256,512),
            torch.nn.BatchNorm1d(512, 0.8),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Linear(512, 1024),
            torch.nn.BatchNorm1d(1024, 0.8),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Linear(1024, 28*28*1),
            torch.nn.Tanh()
        )
    def forward(self, z):
        x= self.G_lay1(z)
        img=x.view(x.size(0),1,28,28)#64*784-》64*1*28*28
        return img

#判别器
class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.D_lay1=torch.nn.Sequential(
            torch.nn.Linear(784,512),
            torch.nn.LeakyReLU(0.2,inplace=True),
            torch.nn.Linear(512, 256),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(256,1),
            torch.nn.Sigmoid()
        )
    def forward(self, img):
        x=img.view(img.size(0),-1)#28*28*1的图转化成784向量
        return self.D_lay1(x)

#实例化
g_net=Generater().cuda()
d_net=Discriminator().cuda()

#损失函数，优化器
loss_fun=torch.nn.BCELoss().cuda()

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
            torchvision.utils.save_image(fake_img.data[:25], "./result_gan_minst/%d.png" % batches_done, nrow=5, normalize=True)











