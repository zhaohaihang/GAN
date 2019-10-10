import torch
import torchvision
import torch.utils.data
import torch.nn
import numpy
import torch.autograd
import torchvision.utils
import itertools
#图像读入与处理
transform=torchvision.transforms.Compose([
    torchvision.transforms.Resize(32),
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

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def to_categorical(y, num_columns):
    """Returns one-hot encoded Variable"""
    y_cat = numpy.zeros((y.shape[0], num_columns))
    y_cat[range(y.shape[0]), y] = 1.0

    return torch.autograd.Variable(torch.cuda.FloatTensor(y_cat))

class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_embedding = torch.nn.Embedding(10, 10)

        self.lay1=torch.nn.Sequential(
            torch.nn.Linear(62+10+2,128*8*8),
        )

        self.lay2=torch.nn.Sequential(

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

    def forward(self, noise,lab,code):
        # lab=self.label_embedding(lab)#64*1->64*10

        x=torch.cat((noise,lab,code),-1)
        x=self.lay1(x)
        x=x.view(x.size(0),128,8,8)
        img=self.lay2(x)
        return img

class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.lay1=torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3, 2, 1),
            torch.nn.LeakyReLU(0.2, inplace=True),
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
            torch.nn.Linear(512,1)#判断是否是生成的图
        )

        self.lay3=torch.nn.Sequential(
            torch.nn.Linear(512,10),
            torch.nn.Softmax()#判断是哪一类
        )
        self.lay4=torch.nn.Sequential(
            torch.nn.Linear(512,2)#隐含信息
        )


    def forward(self, img):
        x=self.lay1(img)
        x=x.view(x.size(0),-1)

        vaildity=self.lay2(x)
        lab=self.lay3(x)
        code=self.lay4(x)

        return vaildity,lab,code

generator = Generator().cuda()
discriminator = Discriminator().cuda()

adversarial_loss = torch.nn.MSELoss().cuda()
categorical_loss = torch.nn.CrossEntropyLoss().cuda()
continuous_loss = torch.nn.MSELoss().cuda()


generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5,0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5,0.999))
optimizer_info = torch.optim.Adam(itertools.chain(generator.parameters(), discriminator.parameters()), lr=0.0002, betas=(0.5,0.999))

epoch_n=20
lambda_cat = 1
lambda_con = 0.1

static_noise_input=torch.autograd.Variable(torch.cuda.FloatTensor(numpy.random.normal(0,1,(10*10,62))))#高斯分布
static_lab_input=to_categorical(numpy.random.randint(0, 10, 10*10), num_columns=10)#热编码
static_code_input=torch.autograd.Variable(torch.cuda.FloatTensor(numpy.random.uniform(-1,1,(10*10,2))))#均匀分布

for epoch in range(epoch_n):
    for i,(real_img,real_lab) in enumerate(test_data_load):
        img_num=real_img.size(0)

        real = torch.autograd.Variable(torch.cuda.FloatTensor(img_num, 1).fill_(1.0), requires_grad=False)
        fake = torch.autograd.Variable(torch.cuda.FloatTensor(img_num, 1).fill_(0.0), requires_grad=False)

        real_img = torch.autograd.Variable(real_img.type(torch.cuda.FloatTensor))

        #G
        optimizer_G.zero_grad()
        noise_input=torch.autograd.Variable(torch.cuda.FloatTensor(numpy.random.normal(0,1,(img_num,62))))#高斯分布
        lab_input=to_categorical(numpy.random.randint(0, 10, img_num), num_columns=10)#热编码
        code_input=torch.autograd.Variable(torch.cuda.FloatTensor(numpy.random.uniform(-1,1,(img_num,2))))#均匀分布

        fake_img=generator(noise_input,lab_input,code_input)

        fake_img_result,_,_=discriminator(fake_img)
        g_loss=adversarial_loss(fake_img_result,real)
        g_loss.backward()
        optimizer_G.step()

        #D
        optimizer_D.zero_grad()

        real_img_result,_,_=discriminator(real_img)
        d_real_loss=adversarial_loss(real_img_result,real)

        fake_img_result, _, _ = discriminator(fake_img.detach())
        d_fake_loss=adversarial_loss(fake_img_result,fake)

        d_loss=(d_fake_loss+d_real_loss)/2
        d_loss.backward()
        optimizer_D.step()

        #info
        optimizer_info.zero_grad()

        temp_lab=numpy.random.randint(0, 10, img_num)

        info_lab=torch.autograd.Variable(torch.cuda.LongTensor(temp_lab), requires_grad=False)

        noise_input = torch.autograd.Variable(torch.cuda.FloatTensor(numpy.random.normal(0, 1, (img_num, 62))))  # 高斯分布
        lab_input = to_categorical(temp_lab, num_columns=10)  # 热编码
        code_input = torch.autograd.Variable(torch.cuda.FloatTensor(numpy.random.uniform(-1, 1, (img_num, 2))))  # 均匀分布

        fake_img=generator(noise_input,lab_input,code_input)
        _, fake_img_lab, fake_img_code = discriminator(fake_img)

        info_loss=lambda_cat*categorical_loss(fake_img_lab,info_lab)+lambda_con*continuous_loss(fake_img_code,code_input)

        info_loss.backward()
        optimizer_info.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [info loss: %f]"
            % (epoch, epoch, i, len(test_data_load), d_loss.item(), g_loss.item(), info_loss.item())
        )
        batches_done = epoch * len(test_data_load) + i
        if batches_done%400==0:
            #static
            noise_input = torch.autograd.Variable(torch.cuda.FloatTensor(numpy.random.normal(0, 1, (10*10, 62))))
            static_sample = generator(noise_input, static_lab_input, static_code_input)
            torchvision.utils.save_image(static_sample.data, "./result_infogan/images_static_%d.png" % batches_done, nrow=10,normalize=True)