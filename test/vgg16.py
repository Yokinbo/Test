


import torchvision


vgg = torchvision.models.vgg16(pretrained=False)  

print(vgg)  # 打印模型结构