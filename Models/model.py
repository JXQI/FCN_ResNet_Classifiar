import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import models
import pretrainedmodels
from efficientnet_pytorch import EfficientNet
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Model:
    def __init__(self,net,class_num=2,pretrained=False):
        self.class_num=class_num
        self.pretrained=pretrained
        self.net=net
    def Net(self):
        if self.net=="ResNet50":
            Net=models.resnet50(pretrained=self.pretrained)
            Net.fc=nn.Linear(2048,self.class_num)
        elif self.net=='DenseNet121':
            Net = models.densenet121(pretrained=True)
            Net.fc=nn.Linear(1024,self.class_num)
        elif self.net=='SENet':
            Net = models.squeezenet1_0(pretrained=True)
            Net.classifier = nn.Sequential(nn.Dropout(p=0.5), nn.Conv2d(512, self.class_num, kernel_size=1), nn.ReLU(inplace=True),
                                           nn.AdaptiveAvgPool2d((1, 1)))
        elif self.net=='pnasnet':
            model_name = 'pnasnet5large'
            if self.pretrained:
                Net = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
            else:
                Net = pretrainedmodels.__dict__[model_name](num_classes=1000)
            Net.last_linear = nn.Linear(4320, 2)
            #Net.eval()
        elif self.net=='efficientNet':
            if self.pretrained:
                Net=EfficientNet.from_pretrained('efficientnet-b0')
            else:
                Net=EfficientNet.from_name('efficientnet-b0')
            Net._fc = nn.Linear(1280, self.class_num, bias=True)
            #Net.eval()
        return Net

if __name__=='__main__':
    model=Model()
    transform = transforms.Compose([transforms.ToTensor()])
    #d = dataloader(path='./data', transforms=transform)
    #feature,label=d[0]
    #测试
    y=model.Net()
    print(y.name)
    print(y.features1)
