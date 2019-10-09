import torch
import torch.nn
import torch.utils.data
import torchvision
import numpy
import torch.autograd
from torchvision.utils import save_image

transforms=torchvision.transforms.Compose([
    torchvision.transforms.Resize(32),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5],[0.5])
])

data_set=torchvision.datasets.MNIST(
    root="./minst_data/",
    train=True,
    download=True,
    transform=transforms
)

data_loader=torch.utils.data.DataLoader(
    dataset=data_set,
    shuffle=True,
    batch_size=64
)

class G(torch.nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.label_emb = torch.nn.Embedding(10, 10)#每一类用10*1向量
        self.model1=torch.nn.Sequential(
            torch.nn.Linear(100+10,128),
            torch.nn.LeakyReLU(0.2,inplace=True),

            torch.nn.Linear(128,256),
            torch.nn.BatchNorm1d(256,0.8),
            torch.nn.LeakyReLU(0.2,inplace=True),

            torch.nn.Linear(256, 512),
            torch.nn.BatchNorm1d(512, 0.8),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Linear(512, 1024),
            torch.nn.BatchNorm1d(1024, 0.8),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.Linear(1024,1*32*32),
            torch.nn.Tanh()
        )
    def forward(self, noise,lable):
        #print(lable.shape)64*1
        lab=self.label_emb(lable)#64*10
        #lab每一行代表一类
        input=torch.cat((lab,noise),-1)#64*110
        img=self.model1(input)#64*1024
        img=img.view(img.size(0),1,32,32)#64*1024->64*1*32*32
        return img


class D(torch.nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.label_embedding = torch.nn.Embedding(10,10)

        self.model = torch.nn.Sequential(
            torch.nn.Linear(10 + 1*32*32, 512),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(512, 512),
            torch.nn.Dropout(0.4),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(512, 512),
            torch.nn.Dropout(0.4),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(512, 1)
        )
    def forward(self, img,label):
        img=img.view(img.size(0), -1)#64,1,32,32  ->   64,1024
        label=self.label_embedding(label)#64*10
        d_in = torch.cat((img,label), -1)#64*1034
        validity = self.model(d_in)
        return validity

loss_fun=torch.nn.MSELoss().cuda()

g_net=G().cuda()
d_net=D().cuda()

optimizer_g=torch.optim.Adam(g_net.parameters(),lr=0.0002,betas=(0.5,0.999))
optimizer_d=torch.optim.Adam(d_net.parameters(),lr=0.0002,betas=(0.5,0.999))

epoch_n=20
ind=0

#在生成网络的输入和鉴别网络的输入都混入label，
# 这样生成网络就会学会根据label生成含有label特征的图片；
# 鉴别网络就能学会根据label快速学会分类图片。
for epoch in range(epoch_n):
    for i,(real_img,real_lab) in enumerate(data_loader):
        img_num=real_img.shape[0]

        real = torch.autograd.Variable(torch.cuda.FloatTensor(img_num, 1).fill_(1.0), requires_grad=False)
        fake = torch.autograd.Variable(torch.cuda.FloatTensor(img_num, 1).fill_(0.0), requires_grad=False)

        real_img=torch.autograd.Variable(real_img.type(torch.cuda.FloatTensor))
        real_lab=torch.autograd.Variable(real_lab.type(torch.cuda.LongTensor))

        #G
        optimizer_g.zero_grad()

        noise=torch.autograd.Variable(torch.cuda.FloatTensor(numpy.random.normal(0,1,(img_num,100))))#randn（）函数使noise服从高斯分布
        fake_lab= torch.autograd.Variable(torch.cuda.LongTensor(numpy.random.randint(0, 10, img_num)))

        #print(fake_lab)

        fake_img=g_net(noise,fake_lab)#由噪声加标签生成样本

        loss=d_net(fake_img,fake_lab)#样本与标签判别

        g_loss=loss_fun(loss,real)
        g_loss.backward()
        optimizer_g.step()

        #_______________________________________________________
        optimizer_d.zero_grad()
        d_real_result=d_net(real_img,real_lab)
        d_real_result_loss=loss_fun(d_real_result,real)

        d_fake_result=d_net(fake_img.detach(), fake_lab)
        d_fake_result_loss=loss_fun(d_fake_result,fake)

        d_loss=(d_fake_result_loss+d_real_result_loss)/2


        d_loss.backward()
        optimizer_d.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, epoch_n, i, len(data_loader), d_loss.item(), g_loss.item())
        )
        ind+=1
        if ind%400==0:
            noise=torch.autograd.Variable(torch.cuda.FloatTensor(numpy.random.normal(0,1,(100,100))))
            labels = numpy.array([num for _ in range(10) for num in range(10)])
            labels = torch.autograd.Variable(torch.cuda.LongTensor(labels))
            gen_imgs = g_net(noise, labels)
            save_image(gen_imgs.data, "./cgan_result_image/image-%d.png" % ind, nrow=10, normalize=True)

