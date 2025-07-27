"""class Parent:
    def __init__(self):
        print("Parent 初始化")

class Child(Parent):
    def __init__(self):
        super().__init__()  # 👈 调用父类构造函数
        print("Child 初始化")
#jishdcilwbjesc
c = Child()
"""
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))