import torch.nn as nn
import torch
import math
from torch.utils import model_zoo
from PIL import Image
import numpy as np

class baseline_class(nn.Module):
    def __init__(self):
        super(baseline_class, self).__init__()
        self.VGG_class = vgg_class()
        # for p in self.parameters():
        #     p.requires_grad = False
        # net = torch.load('/media/wf/WF/dbd/0825/test-32-1025-r=10/211.pth')  # hao fenlei
        # # net = torch.load('/media/weifei/E/pytorch-resnet/0529/test-5556-0724/241.pth')
        # mynet = vgg_class()
        # model_new = mynet.state_dict()
        # for k, v in net.items():
        #     if 'module' in k:
        #         k = k.split('module.')
        #     if 'VGG_class' in k:
        #         k = k.split('VGG_class.')
        #     model_new[k[-1]] = v
        # # self.resnet_DE.load_state_dict(model_new)
        # self.VGG_class.load_state_dict(model_new, strict=False)

    def forward(self, input_1):
        dbd1, class1, s4 = self.VGG_class(input_1)
        return dbd1, class1, s4

class vgg_class(nn.Module):
    def __init__(self):
        super(vgg_class, self).__init__()
        self.base1 = Base1()
        self.base21 = Base21()
        self.base22 = Base22()
        self.base23 = Base23()
        self.base24 = Base24()
        self.output_dbd = Output_dbd()
        self.base41 = Base41()
        self.output_class = Output_class()

    def forward(self, input_1):
        s1, s2, s3, s4,f41 = self.base1(input_1)
        fd1 = self.base21(s4, s3)
        fd2 = self.base22(s2, fd1)
        fd3 = self.base23(s1, fd2)
        fd4 = self.base24(fd3)
        dbd1 = self.output_dbd(fd4)

        fc1 = self.base41(s4)
        class1 = self.output_class(fc1)
        return dbd1, class1, s4

class Base1(nn.Module):
    def __init__(self):
        super(Base1, self).__init__()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv1_1_2 = BaseConv(3, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv1_2_2 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_1_2 = BaseConv(64, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_2_2 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_1_2 = BaseConv(128, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_2_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_3_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_1_2 = BaseConv(256, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_2_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_3_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_1_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_2_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_3_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, x):
        x = self.conv1_1_2(x)
        x = self.conv1_2_2(x)
        x = self.maxpool(x)
        x = self.conv2_1_2(x)
        x = self.conv2_2_2(x)
        s1 = x
        x = self.maxpool(x)
        x = self.conv3_1_2(x)
        x = self.conv3_2_2(x)
        x = self.conv3_3_2(x)
        s2 = x

        x = self.maxpool(x)
        x = self.conv4_1_2(x)
        x = self.conv4_2_2(x)
        x = self.conv4_3_2(x)
        s3 = x
        # print(x.size())
        x = self.maxpool(x)
        x = self.conv5_1_2(x)
        x = self.conv5_2_2(x)
        x = self.conv5_3_2(x)
        s4 = x

        return s1, s2, s3, s4, s4

class Base21(nn.Module):
    def __init__(self):
        super(Base21, self).__init__()
        self.conv1_2 = BaseConv(1024, 256, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
    def forward(self, s4, s3):
        x = s4
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, s3], 1)
        x = self.conv1_2(x)
        x = self.conv2_2(x)
        fd1 = x

        return fd1

class Base22(nn.Module):
    def __init__(self):
        super(Base22, self).__init__()
        self.conv3_2 = BaseConv(512, 128, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_2 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, s2, fd1):
        x = fd1
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, s2], 1)
        x = self.conv3_2(x)
        x = self.conv4_2(x)
        fd2 = x

        return fd2

class Base23(nn.Module):
    def __init__(self):
        super(Base23, self).__init__()
        self.conv5_2 = BaseConv(256, 64, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv6_2 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv7_2 = BaseConv(64, 32, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, s1, fd2):
        x = fd2
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, s1], 1)
        x = self.conv5_2(x)
        x = self.conv6_2(x)
        x = self.conv7_2(x)
        fd3 = x

        return fd3

class Base24(nn.Module):
    def __init__(self):
        super(Base24, self).__init__()
        self.conv_out_base_1 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_out_base_2 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, fd3):
        x = fd3
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv_out_base_1(x)
        x = self.conv_out_base_2(x)
        fd4 = x

        return fd4

class Output_dbd(nn.Module):
    def __init__(self):
        super(Output_dbd, self).__init__()
        self.conv_out_base_3 = BaseConv(32, 1, 1, 1, activation=nn.Sigmoid(), use_bn=False)

    def forward(self, fd4):
        x = fd4
        x = self.conv_out_base_3(x)
        dbd1 = x

        return dbd1

class Base41(nn.Module):
    def __init__(self):
        super(Base41, self).__init__()
        self.conv_class_1 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_class_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, s4):
        x = s4
        x = self.conv_class_1(x)
        x = self.conv_class_2(x)
        fc1 = x

        return fc1

class Output_class(nn.Module):
    def __init__(self):
        super(Output_class, self).__init__()
        self.avgpool = nn.AvgPool2d((20,20))
        self.conv_fc_1 = BaseConv(512, 256, 1, 1, activation=None, use_bn=False)
        self.conv_fc_2 = BaseConv(256, 2, 1, 1, activation=None, use_bn=False)



    def forward(self, fc1):
        x = fc1
        x = self.avgpool(x)
        x = self.conv_fc_1(x)
        x = self.conv_fc_2(x)
        x = x.view(x.size(0), -1)
        class1 = x
        return class1


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, activation=None, use_bn=False):
        super(BaseConv, self).__init__()
        self.use_bn = use_bn
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, kernel // 2)
        self.conv.weight.data.normal_(0, 0.01)
        self.conv.bias.data.zero_()
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, input):
        input = self.conv(input)
        if self.use_bn:
            input = self.bn(input)
        if self.activation:
            input = self.activation(input)

        return input