import torch                                                               #导入包
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential

class Tudui(nn.Module):                       #设计类
    def __init__(self):                       #定义构造函数
        super(Tudui, self).__init__()         #调用父类的构造函数
        self.model1 = Sequential(             #实例化Sequential 类，按顺序组合下列多个神经网络层   
            Conv2d(3, 32, 5, padding=2),      #设计的卷积层，包括其中的各个参数
            MaxPool2d(2),                     #池化层
            Conv2d(32, 32, 5, padding=2),     #第二个卷积层
            MaxPool2d(2),                     #第二个池化层
            Conv2d(32, 64, 5, padding=2),     #第三个卷积层
            MaxPool2d(2),                     #第三个池化层
            Flatten(),                        #展平层，将多维输入一维化
            Linear(1024, 64),                 #全连接层，输入为1024维，输出为64维
            Linear(64, 10)                    #全连接层，输入为64维，输出为10维
        )

    def forward(self, x):                     #定义前向传播函数
        x = self.model1(x)                    #将输入x传入模型model1中
        return x                              #返回输出x

tudui = Tudui()                               #实例化Tudui类
print(tudui)                                  #打印模型结构


input = torch.ones((64, 3, 32, 32))           #创建一个输入张量，形状为(64, 3, 32, 32)，表示64个样本，每个样本有3个通道，大小为32x32  
output = tudui(input)                         #将输入张量传入模型，得到输出张量
print(output.shape)                           #打印输出张量的形状，应该是(64, 10)，表示64个样本，每个样本有10个输出类别

print(dir(tudui))
