import torch.nn as nn
import torch
import math
from baseline_vgg_class import *
from torch.utils import model_zoo
from PIL import Image
import numpy as np

class resnet_dbd_edge(nn.Module):
    def __init__(self):
        super(resnet_dbd_edge, self).__init__()
        self.baseline_class = vgg_class()
        self.resnet_DE_main = resnet_d1_e1()
        self.avgpool = nn.AvgPool2d((320, 320))
        self.avgpool2 = nn.AvgPool2d((20,20))
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(512, 512)
        self.relu = nn.ReLU(inplace = True)
    def forward(self, input_1):
        dbd1, edge1, f41, f51 = self.resnet_DE_main(input_1) #hun
        return dbd1, edge1

class resnet_d1_e1(nn.Module):
    def __init__(self):
        super(resnet_d1_e1, self).__init__()
        self.base1 = Base1()
        self.fill_hollow = SFAoatt()
        self.base3 = BaseEdge()
        self.loukong = SFAoatt2()
        self.base2 = BaseDBD()

        self.base21 = Base21()
        self.base22 = Base22()
        self.base23 = Base23()
        self.base24 = Base24()
        self.base31 = Base31()
        self.base32 = Base32()
        self.base33 = Base33()
        self.base34 = Base34()
        self.lff1 = LFF1()
        self.tff2 = TFF2()
        self.lff2 = LFF2()
        self.lff3 = LFF3()
        self.tff4 = TFF4()
        self.addff1 = AddFF1()
        self.addff2 = AddFF2()
        self.addff3 = AddFF3()
        self.addff4 = AddFF4()
        self.output_dbd = Output_dbd()
        self.output_edge = Output_edge()
        self.base41 = Base41()
        self.output_class = Output_class()



    def forward(self, input_1):
        s1, s2, s3, s4, f41, f51 = self.base1(input_1)
        fe1 = self.base31(f51, s3)
        fe2 = self.base32(s2, fe1)
        fd1 = self.base21(f51, s3)
        fd2 = self.base22(s2, fd1)
        fe3 = self.base33(s1, fe2)
        fe4 = self.base34(fe3)
        edge1 = self.output_edge(fe4)
        fd3 = self.base23(s1, fd2)
        fd4 = self.base24(fd3)
        dbd1 = self.output_dbd(fd4)
        class1 = self.base41(f51)
        class1 = self.output_class(class1)
        return dbd1, edge1, f41, f51


class Base1(nn.Module):
    def __init__(self):
        super(Base1, self).__init__()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d((20, 20))
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
        self.conv6_1_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv6_2_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv6_3_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv7_1_2 = BaseConv(1024, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv7_2_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv7_3_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv8_1_2 = BaseConv(1024, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv8_2_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv8_3_2 = BaseConv(512, 1, 3, 1, activation=nn.Sigmoid(), use_bn=False)
        self.conv9_1_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv9_2_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv9_3_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv10_1_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv10_2_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv10_3_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv11_1_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv11_2_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv11_3_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv12_1_2 = BaseConv(1024, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv12_2_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv12_3_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv13_1_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv13_2_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv13_3_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)

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
        x = self.maxpool(x)
        x = self.conv5_1_2(x)
        x = self.conv5_2_2(x)
        x = self.conv5_3_2(x)
        s4 = x
        f51 = x

        x = s3
        x = self.maxpool(x)
        x = self.conv6_1_2(x)
        x = self.conv6_2_2(x)
        x = self.conv6_3_2(x)
        f41 = x

        channel1 = torch.cat([f41, f51], 1)
        channel1 = self.conv12_1_2(channel1)
        channel1 = self.conv12_2_2(channel1)
        channel1 = self.conv12_3_2(channel1)
        channel1 = self.avgpool(channel1)
        fch = torch.mul(f41, channel1)
        fch = self.conv13_1_2(fch)
        fch = self.conv13_2_2(fch)
        fch = self.conv13_3_2(fch)
        f51 = torch.add(f51, fch)
        return s1, s2, s3, s4, f41, f51

class AddFF1(nn.Module):
    def __init__(self):
        super(AddFF1, self).__init__()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.down_ff1 = BaseConv(32, 512, 1, 1, activation=nn.ReLU(), use_bn=True)
    def forward(self, fe3, s4):
        fd = s4
        fe = fe3
        fe = self.down_ff1(fe)
        fe = self.maxpool(fe)
        fe = self.maxpool(fe)
        fe = self.maxpool(fe)
        fe1f = torch.add(fe, fd)

        return fe1f

class AddFF2(nn.Module):
    def __init__(self):
        super(AddFF2, self).__init__()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.down_ff2 = BaseConv(32, 256, 1, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, fe3, fd1):
        fd = fd1
        fe = fe3
        fe = self.down_ff2(fe)
        fe = self.maxpool(fe)
        fe = self.maxpool(fe)
        fe2f = torch.add(fe, fd)

        return fe2f

class AddFF3(nn.Module):
    def __init__(self):
        super(AddFF3, self).__init__()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.down_ff3 = BaseConv(32, 128, 1, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, fe2, fd2):
        fd = fd2
        fe = fe2
        fe3f = torch.add(fe, fd)

        return fe3f

class AddFF4(nn.Module):
    def __init__(self):
        super(AddFF4, self).__init__()


    def forward(self, fe3, fd3):
        fd = fd3
        fe = fe3
        fe4f = torch.add(fe, fd)

        return fe4f

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

class Base31(nn.Module):
    def __init__(self):
        super(Base31, self).__init__()
        self.conv1_2 = BaseConv(1024, 256, 1, 1, activation=nn.ReLU(), use_bn=True)
        # self.conv1_2 = BaseConv(512, 256, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
    def forward(self, s4, s3):
    # def forward(self, s4):
        x = s4
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, s3], 1)
        x = self.conv1_2(x)
        x = self.conv2_2(x)
        fe1 = x

        return fe1

class Base411(nn.Module):
    def __init__(self):
        super(Base411, self).__init__()
        self.conv1_2 = BaseConv(512, 256, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
    def forward(self, s4):
        x = s4
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv1_2(x)
        x = self.conv2_2(x)
        fm1 = x

        return fm1

class LFF1(nn.Module):
    def __init__(self):
        super(LFF1, self).__init__()
        self.conv_fff_0_1 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_0_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_0_3 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_0_4 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_0_5 = BaseConv(256, 1, 1, 1, activation=nn.Sigmoid(), use_bn=False)

        self.conv_fff_1_1 = BaseConv(512, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_1_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_1_3 = BaseConv(256, 1, 3, 1, activation=nn.Sigmoid(), use_bn=False)

        self.conv_fff_1_4 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_1_5 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_1_6 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, fd1, fe1):
        fe = fe1
        fd = fd1
        fd = self.conv_fff_0_1(fd)
        fd = self.conv_fff_0_2(fd)
        fd = self.conv_fff_0_3(fd)
        edge_ff1 = self.conv_fff_0_4(fd)
        edge_ff1 = self.conv_fff_0_5(edge_ff1)

        fmap = torch.cat([fe, fd], 1)
        fmap = self.conv_fff_1_1(fmap)
        fmap = self.conv_fff_1_2(fmap)
        fmap = self.conv_fff_1_3(fmap)

        fcoo = torch.mul(fd, fmap)
        fdif = torch.sub(fd, fcoo)
        fdif = self.conv_fff_1_4(fdif)
        fdif = self.conv_fff_1_5(fdif)
        fdif = self.conv_fff_1_6(fdif)

        fe1f = torch.add(fe, fdif)

        return fe1f, edge_ff1

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

class Base32(nn.Module):
    def __init__(self):
        super(Base32, self).__init__()
        self.conv3_2 = BaseConv(512, 128, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_2 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, s2, fe1f):
        x = fe1f
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, s2], 1)
        x = self.conv3_2(x)
        x = self.conv4_2(x)
        fe2 = x

        return fe2

class Base42(nn.Module):
    def __init__(self):
        super(Base42, self).__init__()
        self.conv3_2 = BaseConv(256, 128, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_2 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, fe1f):
        x = fe1f
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv3_2(x)
        x = self.conv4_2(x)
        fm2 = x

        return fm2

class TFF2(nn.Module):
    def __init__(self):
        super(TFF2, self).__init__()
        self.conv_fff_0_1 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_0_2 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_0_3 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_0_4 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_0_5 = BaseConv(128, 1, 1, 1, activation=nn.Sigmoid(), use_bn=False)

        self.conv_fff_1_1 = BaseConv(256, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_1_2 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_1_3 = BaseConv(128, 1, 3, 1, activation=nn.Sigmoid(), use_bn=False)

        self.conv_fff_1_4 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_1_5 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_1_6 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, fd2, fe2):
        fe = fe2
        fd = fd2
        fe = self.conv_fff_0_1(fe)
        fe = self.conv_fff_0_2(fe)
        fe = self.conv_fff_0_3(fe)

        fmap = torch.cat([fd, fe], 1)
        fmap = self.conv_fff_1_1(fmap)
        fmap = self.conv_fff_1_2(fmap)
        fmap = self.conv_fff_1_3(fmap)

        fcoo = torch.mul(fe, fmap)
        fd2f = torch.add(fd, fcoo)

        return fd2f

class LFF2(nn.Module):
    def __init__(self):
        super(LFF2, self).__init__()
        self.conv_fff_0_1 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_0_2 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_0_3 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv_fff_0_4 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_0_5 = BaseConv(128, 1, 1, 1, activation=nn.Sigmoid(), use_bn=False)

        self.conv_fff_1_1 = BaseConv(256, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_1_2 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_1_3 = BaseConv(128, 1, 3, 1, activation=nn.Sigmoid(), use_bn=False)

        self.conv_fff_1_4 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_1_5 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_1_6 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, fd2, fe2):
        fe = fe2
        fd = fd2
        fe = self.conv_fff_0_1(fe)
        fe = self.conv_fff_0_2(fe)
        fe = self.conv_fff_0_3(fe)

        fmap = torch.cat([fd, fe], 1)
        fmap = self.conv_fff_1_1(fmap)
        fmap = self.conv_fff_1_2(fmap)
        fmap = self.conv_fff_1_3(fmap)

        fcoo = torch.mul(fe, fmap)
        fd2f = torch.add(fd, fcoo)

        return fd2f

class Base23(nn.Module):
    def __init__(self):
        super(Base23, self).__init__()
        self.conv5_2 = BaseConv(256, 64, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv6_2 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv7_2 = BaseConv(64, 32, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, s1, fd2f):
        x = fd2f
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, s1], 1)
        x = self.conv5_2(x)
        x = self.conv6_2(x)
        x = self.conv7_2(x)
        fd3 = x

        return fd3

class Base33(nn.Module):
    def __init__(self):
        super(Base33, self).__init__()
        self.conv5_2 = BaseConv(256, 64, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv6_2 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv7_2 = BaseConv(64, 32, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, s1, fe2):
        x = fe2
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, s1], 1)
        x = self.conv5_2(x)
        x = self.conv6_2(x)
        x = self.conv7_2(x)
        fe3 = x

        return fe3

class Base43(nn.Module):
    def __init__(self):
        super(Base43, self).__init__()
        # self.conv5_2 = BaseConv(256, 64, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_2 = BaseConv(128, 64, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv6_2 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv7_2 = BaseConv(64, 32, 3, 1, activation=nn.ReLU(), use_bn=True)

    # def forward(self, s1, fe2):
    def forward(self, fe2):
        x = fe2
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        # x = torch.cat([x, s1], 1)
        x = self.conv5_2(x)
        x = self.conv6_2(x)
        x = self.conv7_2(x)
        fm3 = x

        return fm3

class LFF3(nn.Module):
    def __init__(self):
        super(LFF3, self).__init__()
        ##FFF3
        self.conv_fff_0_1 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_0_2 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_0_3 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        # 监督edge
        self.conv_fff_0_4 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_0_5 = BaseConv(32, 1, 1, 1, activation=nn.Sigmoid(), use_bn=False)

        self.conv_fff_1_1 = BaseConv(64, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_1_2 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_1_3 = BaseConv(32, 1, 3, 1, activation=nn.Sigmoid(), use_bn=False)

        self.conv_fff_1_4 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_1_5 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_1_6 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, fd3, fe3):
        ##FFF3
        fe = fe3
        fd = fd3
        fd = self.conv_fff_0_1(fd)
        fd = self.conv_fff_0_2(fd)
        fd = self.conv_fff_0_3(fd)
        edge_ff3 = self.conv_fff_0_4(fd)
        edge_ff3 = self.conv_fff_0_5(edge_ff3)

        fmap = torch.cat([fe, fd], 1)
        fmap = self.conv_fff_1_1(fmap)
        fmap = self.conv_fff_1_2(fmap)
        fmap = self.conv_fff_1_3(fmap)

        fcoo = torch.mul(fd, fmap)
        fdif = torch.sub(fd, fcoo)
        fdif = self.conv_fff_1_4(fdif)
        fdif = self.conv_fff_1_5(fdif)
        fdif = self.conv_fff_1_6(fdif)

        fe3f = torch.add(fe, fdif)

        return fe3f, edge_ff3

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

class Base34(nn.Module):
    def __init__(self):
        super(Base34, self).__init__()
        self.conv_out_base_1 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_out_base_2 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, fe3f):
        x = fe3f
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv_out_base_1(x)
        x = self.conv_out_base_2(x)
        fe4 = x

        return fe4

class Base44(nn.Module):
    def __init__(self):
        super(Base44, self).__init__()
        self.conv_out_base_1 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_out_base_2 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, fe3f):
        x = fe3f
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv_out_base_1(x)
        x = self.conv_out_base_2(x)
        fm4 = x

        return fm4

class TFF4(nn.Module):
    def __init__(self):
        super(TFF4, self).__init__()
        ##FFF4
        self.conv_fff_0_1 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_0_2 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_0_3 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        # 监督dbd
        self.conv_fff_0_4 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_0_5 = BaseConv(32, 1, 1, 1, activation=nn.Sigmoid(), use_bn=False)

        self.conv_fff_1_1 = BaseConv(64, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_1_2 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_1_3 = BaseConv(32, 1, 3, 1, activation=nn.Sigmoid(), use_bn=False)

        self.conv_fff_1_4 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_1_5 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_1_6 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, fd4, fe4):
        ##FFF4
        fe = fe4
        fd = fd4
        fe = self.conv_fff_0_1(fe)
        fe = self.conv_fff_0_2(fe)
        fe = self.conv_fff_0_3(fe)
        dbd_ff4 = self.conv_fff_0_4(fe)
        dbd_ff4 = self.conv_fff_0_5(dbd_ff4)

        fmap = torch.cat([fd, fe], 1)
        fmap = self.conv_fff_1_1(fmap)
        fmap = self.conv_fff_1_2(fmap)
        fmap = self.conv_fff_1_3(fmap)

        fcoo = torch.mul(fe, fmap)
        fdif = torch.sub(fe, fcoo)
        fdif = self.conv_fff_1_4(fdif)
        fdif = self.conv_fff_1_5(fdif)
        fdif = self.conv_fff_1_6(fdif)

        fd4f = torch.add(fd, fdif)

        return fd4f, dbd_ff4

class Base41(nn.Module):
    def __init__(self):
        super(Base41, self).__init__()
        # self.maxpool = nn.MaxPool2d(2, 2)
        self.conv_class_1 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_class_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        # self.conv_class_3 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, s4):
        x = s4
        # x = self.maxpool(x)
        x = self.conv_class_1(x)
        x = self.conv_class_2(x)
        # x = self.conv_class_3(x)
        fc1 = x
        # print(x.size)

        return fc1

class Output_dbd(nn.Module):
    def __init__(self):
        super(Output_dbd, self).__init__()
        self.conv_out_base_3 = BaseConv(32, 1, 1, 1, activation=nn.Sigmoid(), use_bn=False)

    def forward(self, fd4f):
        x = fd4f
        # print(x.size())
        x = self.conv_out_base_3(x)
        # print(x.size())
        dbd1 = x

        return dbd1

class Output_edge(nn.Module):
    def __init__(self):
        super(Output_edge, self).__init__()
        self.conv_out_base_3 = BaseConv(32, 1, 1, 1, activation=nn.Sigmoid(), use_bn=False)

    def forward(self, fe4):
        x = fe4
        x = self.conv_out_base_3(x)
        edge1 = x

        return edge1

class Output_map(nn.Module):
    def __init__(self):
        super(Output_map, self).__init__()
        self.conv_out_base_3 = BaseConv(32, 1, 1, 1, activation=nn.Sigmoid(), use_bn=False)

    def forward(self, fe4):
        x = fe4
        x = self.conv_out_base_3(x)
        map1 = x

        return map1

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
        # print(x.size())
        x = x.view(x.size(0), -1)
        class1 = x
        # print(class1.size())

        return class1

class BaseDBD(nn.Module):
    def __init__(self):
        super(BaseDBD, self).__init__()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv3_1_2 = BaseConv(128, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_2_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_3_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_1_2 = BaseConv(256, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_2_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_3_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_1_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_2_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_3_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv1_2 = BaseConv(1024, 256, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv3_2 = BaseConv(512, 128, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_2 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv5_2 = BaseConv(256, 64, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv6_2 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv7_2 = BaseConv(64, 32, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv_out_base_1 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_out_base_2 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_out_base_3 = BaseConv(32, 1, 1, 1, activation=nn.Sigmoid(), use_bn=False)

        ##FFF1
        self.conv_fff_1_1 = BaseConv(512, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_1_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_1_3 = BaseConv(256, 1, 3, 1, activation=nn.Sigmoid(), use_bn=True)

        self.conv_fff_1_4 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_1_5 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_1_6 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)


        ##FFF2
        self.conv_fff_2_1 = BaseConv(256, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_2_2 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_2_3 = BaseConv(128, 1, 3, 1, activation=nn.Sigmoid(), use_bn=True)

        self.conv_fff_2_4 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_2_5 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_2_6 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)


        ##FFF3
        self.conv_fff_3_1 = BaseConv(64, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_3_2 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_3_3 = BaseConv(32, 1, 3, 1, activation=nn.Sigmoid(), use_bn=True)

        self.conv_fff_3_4 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_3_5 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_3_6 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)


        ##FFF4
        self.conv_fff_4_1 = BaseConv(64, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_4_2 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_4_3 = BaseConv(32, 1, 3, 1, activation=nn.Sigmoid(), use_bn=True)

        self.conv_fff_4_4 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_4_5 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_4_6 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)

        # for p in self.parameters():
        #     p.requires_grad = False

    def forward(self, s1, s2, s3, s4):
        # s1 = x
        # x = self.maxpool(x)
        # x = self.conv3_1_2(x)
        # x = self.conv3_2_2(x)
        # x = self.conv3_3_2(x)
        # s2 = x
        #
        # x = self.maxpool(x)
        # x = self.conv4_1_2(x)
        # x = self.conv4_2_2(x)
        # x = self.conv4_3_2(x)
        # s3 = x
        #
        # x = self.maxpool(x)
        # x = self.conv5_1_2(x)
        # x = self.conv5_2_2(x)
        # x = self.conv5_3_2(x)

        x = s4
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, s3], 1)
        x = self.conv1_2(x)
        x = self.conv2_2(x)
        #
        # ##FFF1
        # ffp2 = fp2
        # ffp2 = torch.cat([x, ffp2], 1)
        # ffp2 = self.conv_fff_1_1(ffp2)
        # ffp2 = self.conv_fff_1_2(ffp2)
        # ffp2 = self.conv_fff_1_3(ffp2)
        # a = ffp2.squeeze()
        # a = a.cpu()
        # a = a.detach().numpy()
        # # print(a)
        # a = a*255
        # a = Image.fromarray(a.astype(np.uint8))
        # a.show()
        # ffp2 = torch.mul(fp2, ffp2)
        # fp2 = torch.sub(fp2, ffp2)
        # fp2 = self.conv_fff_1_4(fp2)
        # fp2 = self.conv_fff_1_5(fp2)
        # fp2 = self.conv_fff_1_6(fp2)
        # # b = fp2.mean(1)
        # # print(b)
        # # b = b.squeeze()
        # # b = b.cpu()
        # # b = b.detach().numpy()
        # # print(b)
        # # b = b*255
        # # b = Image.fromarray(b.astype(np.uint8))
        # # b.show()
        # x = torch.add(fp2, x)

        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, s2], 1)
        x = self.conv3_2(x)
        x = self.conv4_2(x)

        # ##FFF2
        # ffp3 = fp3
        # ffp3 = torch.cat([x, ffp3], 1)
        # ffp3 = self.conv_fff_2_1(ffp3)
        # ffp3 = self.conv_fff_2_2(ffp3)
        # ffp3 = self.conv_fff_2_3(ffp3)
        # ffp3 = torch.mul(fp3, ffp3)
        # fp3 = torch.sub(fp3, ffp3)
        # fp3 = self.conv_fff_2_4(fp3)
        # fp3 = self.conv_fff_2_5(fp3)
        # fp3 = self.conv_fff_2_6(fp3)
        # x = torch.add(fp3, x)

        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, s1], 1)
        x = self.conv5_2(x)
        x = self.conv6_2(x)
        x = self.conv7_2(x)
        #
        # ##FFF3
        # ffp4 = fp4
        # ffp4 = torch.cat([x, ffp4], 1)
        # ffp4 = self.conv_fff_3_1(ffp4)
        # ffp4 = self.conv_fff_3_2(ffp4)
        # ffp4 = self.conv_fff_3_3(ffp4)
        # ffp4 = torch.mul(fp4, ffp4)
        # fp4 = torch.sub(fp4, ffp4)
        # fp4 = self.conv_fff_3_4(fp4)
        # fp4 = self.conv_fff_3_5(fp4)
        # fp4 = self.conv_fff_3_6(fp4)
        # x = torch.add(fp4, x)

        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv_out_base_1(x)
        x = self.conv_out_base_2(x)
        # ##FFF4
        # ffp5 = fp5
        # ffp5 = torch.cat([x, ffp5], 1)
        # ffp5 = self.conv_fff_4_1(ffp5)
        # ffp5 = self.conv_fff_4_2(ffp5)
        # ffp5 = self.conv_fff_4_3(ffp5)
        # ffp5 = torch.mul(fp5, ffp5)
        # fp5 = torch.sub(fp5, ffp5)
        # fp5 = self.conv_fff_4_4(fp5)
        # fp5 = self.conv_fff_4_5(fp5)
        # fp5 = self.conv_fff_4_6(fp5)
        # x = torch.add(fp5, x)

        x = self.conv_out_base_3(x)

        return x

class BaseEdge(nn.Module):
    def __init__(self):
        super(BaseEdge, self).__init__()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv3_1_2 = BaseConv(128, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_2_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_3_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_1_2 = BaseConv(256, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_2_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_3_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_1_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_2_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_3_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv1_2 = BaseConv(1024, 256, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv3_2 = BaseConv(512, 128, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_2 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv5_2 = BaseConv(256, 64, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv6_2 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv7_2 = BaseConv(64, 32, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv_out_base_1 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_out_base_2 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_out_base_3 = BaseConv(32, 1, 1, 1, activation=nn.Sigmoid(), use_bn=False)

        ##FF1
        self.conv_ff_1_1 = BaseConv(512, 256, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.inplanes = 256
        self.res_stack_1 = self.make_stack(256, 2, stride=1)
        self.conv_ff_1_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        ##FF2
        self.conv_ff_2_1 = BaseConv(256, 128, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.res_stack_2 = self.make_stack(128, 2, stride=1)
        self.conv_ff_2_2 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        ##FF3
        self.conv_ff_3_1 = BaseConv(64, 32, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.res_stack_3 = self.make_stack(32, 2, stride=1)
        self.conv_ff_3_2 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        ##FF4
        self.conv_ff_4_1 = BaseConv(64, 32, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.res_stack_4 = self.make_stack(32, 2, stride=1)
        self.conv_ff_4_2 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        # for p in self.parameters():
        #     p.requires_grad = False
        ##FFF1
        self.conv_fff_1_1 = BaseConv(512, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_1_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_1_3 = BaseConv(256, 1, 3, 1, activation=nn.Sigmoid(), use_bn=True)

        self.conv_fff_1_4 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_1_5 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_1_6 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv_fff_1_7 = BaseConv(512, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_1_8 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_1_9 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)


        ##FFF2
        self.conv_fff_2_1 = BaseConv(256, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_2_2 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_2_3 = BaseConv(128, 1, 3, 1, activation=nn.Sigmoid(), use_bn=True)

        self.conv_fff_2_4 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_2_5 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_2_6 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv_fff_2_7 = BaseConv(256, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_2_8 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_2_9 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)

        ##FFF3
        self.conv_fff_3_1 = BaseConv(64, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_3_2 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_3_3 = BaseConv(32, 1, 3, 1, activation=nn.Sigmoid(), use_bn=True)

        self.conv_fff_3_4 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_3_5 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_3_6 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv_fff_3_7 = BaseConv(64, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_3_8 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_3_9 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)

        ##FFF4
        self.conv_fff_4_1 = BaseConv(64, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_4_2 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_4_3 = BaseConv(32, 1, 3, 1, activation=nn.Sigmoid(), use_bn=True)

        self.conv_fff_4_4 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_4_5 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_4_6 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv_fff_4_7 = BaseConv(64, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_4_8 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_fff_4_9 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)

    def init_param(self):
        # The following is initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.shape[0] * m.weight.shape[1]
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
    def make_stack(self, planes, blocks, stride=1):
        downsample = None
        layers = []
        # print(self.inplanes)
        if stride != 1 or self.inplanes != planes * Bottleneck4.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck4.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck4.expansion),
            )

        layers.append(Bottleneck4(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * Bottleneck4.expansion
        for i in range(1, blocks):
            layers.append(Bottleneck4(self.inplanes, planes))

        if self.inplanes/128 > 1:
            self.inplanes = int(self.inplanes/2)
        elif self.inplanes == 128:
            self.inplanes =int(self.inplanes/4)

        return nn.Sequential(*layers)

    def forward(self, s1, s2, s3, s4):
        # s1 = x
        # x = self.maxpool(x)
        # x = self.conv3_1_2(x)
        # x = self.conv3_2_2(x)
        # x = self.conv3_3_2(x)
        # s2 = x
        #
        # x = self.maxpool(x)
        # x = self.conv4_1_2(x)
        # x = self.conv4_2_2(x)
        # x = self.conv4_3_2(x)
        # s3 = x
        #
        # x = self.maxpool(x)
        # x = self.conv5_1_2(x)
        # x = self.conv5_2_2(x)
        # x = self.conv5_3_2(x)
        x = s4
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, s3], 1)
        x = self.conv1_2(x)
        x = self.conv2_2(x)
        ##FF1
        # fp2 = torch.cat([x, fp2], 1)
        # fp2 = self.conv_ff_1_1(fp2)
        # fp2 = self.res_stack_1(fp2)
        # fp2 = self.conv_ff_1_2(fp2)
        # x = torch.add(fp2, x)

        # ##FFF1
        # ffp2 = fp2
        # ffp2 = torch.cat([x, ffp2], 1)
        # ffp2 = self.conv_fff_1_1(ffp2)
        # ffp2 = self.conv_fff_1_2(ffp2)
        # ffp2 = self.conv_fff_1_3(ffp2)
        # # a = ffp2.squeeze()
        # # a = a.cpu()
        # # a = a.detach().numpy()
        # # # print(a)
        # # a = a*255
        # # a = Image.fromarray(a.astype(np.uint8))
        # # a.show()
        # ffp2 = torch.mul(fp2, ffp2)
        # fp2 = torch.sub(fp2, ffp2)
        # fp2 = self.conv_fff_1_4(fp2)
        # fp2 = self.conv_fff_1_5(fp2)
        # fp2 = self.conv_fff_1_6(fp2)
        #
        #
        # # b = fp2.mean(1)
        # # print(b)
        # # b = b.squeeze()
        # # b = b.cpu()
        # # b = b.detach().numpy()
        # # print(b)
        # # b = b*255
        # # b = Image.fromarray(b.astype(np.uint8))
        # # b.show()
        #
        #
        # # x = torch.cat([fp2, x], 1)
        # x = torch.add(fp2,x)
        # # x = self.conv_fff_1_7(x)
        # # x = self.conv_fff_1_8(x)
        # # x = self.conv_fff_1_9(x)

        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, s2], 1)
        x = self.conv3_2(x)
        x = self.conv4_2(x)
        # ##FF2
        # fp3 = torch.cat([x, fp3], 1)
        # fp3 = self.conv_ff_2_1(fp3)
        # fp3 = self.res_stack_2(fp3)
        # fp3 = self.conv_ff_2_2(fp3)
        # x = torch.add(fp3, x)

        # ##FFF2
        # ffp3 = fp3
        # ffp3 = torch.cat([x, ffp3], 1)
        # ffp3 = self.conv_fff_2_1(ffp3)
        # ffp3 = self.conv_fff_2_2(ffp3)
        # ffp3 = self.conv_fff_2_3(ffp3)
        # ffp3 = torch.mul(fp3, ffp3)
        # fp3 = torch.sub(fp3, ffp3)
        # fp3 = self.conv_fff_2_4(fp3)
        # fp3 = self.conv_fff_2_5(fp3)
        # fp3 = self.conv_fff_2_6(fp3)
        # # x = torch.cat([fp3, x], 1)
        # x = torch.add(fp3,x)
        # # x = self.conv_fff_2_7(x)
        # # x = self.conv_fff_2_8(x)
        # # x = self.conv_fff_2_9(x)

        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, s1], 1)
        x = self.conv5_2(x)
        x = self.conv6_2(x)
        x = self.conv7_2(x)
        # ##FF3
        # fp4 = torch.cat([x, fp4], 1)
        # fp4 = self.conv_ff_3_1(fp4)
        # fp4 = self.res_stack_3(fp4)
        # fp4 = self.conv_ff_3_2(fp4)
        # x = torch.add(fp4, x)

        # ##FFF3
        # ffp4 = fp4
        # ffp4 = torch.cat([x, ffp4], 1)
        # ffp4 = self.conv_fff_3_1(ffp4)
        # ffp4 = self.conv_fff_3_2(ffp4)
        # ffp4 = self.conv_fff_3_3(ffp4)
        # ffp4 = torch.mul(fp4, ffp4)
        # fp4 = torch.sub(fp4, ffp4)
        # fp4 = self.conv_fff_3_4(fp4)
        # fp4 = self.conv_fff_3_5(fp4)
        # fp4 = self.conv_fff_3_6(fp4)
        # # x = torch.cat([fp4, x], 1)
        # x = torch.add(fp4,x)
        # # x = self.conv_fff_3_7(x)
        # # x = self.conv_fff_3_8(x)
        # # x = self.conv_fff_3_9(x)

        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv_out_base_1(x)
        x = self.conv_out_base_2(x)
        # ##FF4
        # fp5 = torch.cat([x, fp5], 1)
        # fp5 = self.conv_ff_4_1(fp5)
        # fp5 = self.res_stack_4(fp5)
        # fp5 = self.conv_ff_4_2(fp5)
        # x = torch.add(fp5, x)

        # ##FFF4
        # ffp5 = fp5
        # ffp5 = torch.cat([x, ffp5], 1)
        # ffp5 = self.conv_fff_4_1(ffp5)
        # ffp5 = self.conv_fff_4_2(ffp5)
        # ffp5 = self.conv_fff_4_3(ffp5)
        # ffp5 = torch.mul(fp5, ffp5)
        # fp5 = torch.sub(fp5, ffp5)
        # fp5 = self.conv_fff_4_4(fp5)
        # fp5 = self.conv_fff_4_5(fp5)
        # fp5 = self.conv_fff_4_6(fp5)
        # # x = torch.cat([fp5, x], 1)
        # x = torch.add(fp5, x)
        # # x = self.conv_fff_4_7(x)
        # # x = self.conv_fff_4_8(x)
        # # x = self.conv_fff_4_9(x)

        x = self.conv_out_base_3(x)

        return x

class Bottleneck4(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck4, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out

# 共享stage12net
class ResNet1(nn.Module):

    def __init__(self, layers=[3, 4, 6, 3], model_path="model.pkl"):
        super(ResNet1, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stack1 = self.make_stack(64, layers[0], stride=1)
        self.stack2 = self.make_stack(128, layers[1], stride=2)
        for p in self.parameters():
            p.requires_grad = False

        self.init_param()

    def init_param(self):
        # The following is initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.shape[0] * m.weight.shape[1]
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def make_stack(self, planes, blocks, stride=1):
        downsample = None
        layers = []

        if stride != 1 or self.inplanes != planes * Bottleneck1.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck1.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck1.expansion),
            )

        layers.append(Bottleneck1(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * Bottleneck1.expansion
        for i in range(1, blocks):
            layers.append(Bottleneck1(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.stack1(x)
        x = self.stack2(x)

        return x
class Bottleneck1(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck1, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# dbd1-net
class ResNet2(nn.Module):
    """
    block: A sub module
    """

    def __init__(self, layers=[3, 4, 6, 3]):
        super(ResNet2, self).__init__()
        self.inplanes = 512

        self.stack3 = self.make_stack(256, layers[2], stride=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv_ff3_1 = BaseConv(1056,1024,1,1,activation=nn.ReLU(),use_bn=True)
        self.res_stack1 = self.make_stacks(1024, 2, stride=1)
        self.conv_ff3_2 = BaseConv(1024,1024,3,1,activation=nn.ReLU(),use_bn=True)

        self.stack4 = self.make_stack(512, layers[3], stride=1)

        self.conv_ff4_1 = BaseConv(2080,2048,1,1,activation=nn.ReLU(),use_bn=True)
        self.res_stack2 = self.make_stacks(2048, 2, stride=1)
        self.conv_ff4_2 = BaseConv(2048,2048,3,1,activation=nn.ReLU(),use_bn=True)

        self.conv_down1 = nn.Conv2d(2048, 1, kernel_size=1, stride=1, padding=0,
                                    bias=False)
        self.sigmoid = nn.Sigmoid()
        # initialize parameters
        # for p in self.parameters():
        #     p.requires_grad = False
        self.init_param()

    def init_param(self):
        # The following is initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.shape[0] * m.weight.shape[1]
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def make_stack(self, planes, blocks, stride=1):
        downsample = None
        layers = []

        if stride != 1 or self.inplanes != planes * Bottleneck2.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck2.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck2.expansion),
            )

        layers.append(Bottleneck2(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * Bottleneck2.expansion
        for i in range(1, blocks):
            layers.append(Bottleneck2(self.inplanes, planes))

        return nn.Sequential(*layers)

    def make_stacks(self, planes, blocks, stride=1):
        downsample = None
        layers = []

        if stride != 1 or self.inplanes != planes * Bottleneck3.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck3.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck3.expansion),
            )

        layers.append(Bottleneck3(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * Bottleneck3.expansion
        for i in range(1, blocks):
            layers.append(Bottleneck3(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, fp4, fp5):

        x = self.stack3(x)

        fp4 = self.maxpool(fp4)
        fp4 = self.maxpool(fp4)
        fp4 = self.maxpool(fp4)
        fp4 = torch.cat([x,fp4],1)
        fp4 = self.conv_ff3_1(fp4)
        fp4 = self.res_stack1(fp4)
        fp4 = self.conv_ff3_2(fp4)

        x = x+fp4

        x = self.stack4(x)

        fp5 = self.maxpool(fp5)
        fp5 = self.maxpool(fp5)
        fp5 = self.maxpool(fp5)
        fp5 = torch.cat([x,fp5],1)
        fp5 = self.conv_ff4_1(fp5)
        fp5 = self.res_stack2(fp5)
        fp5 = self.conv_ff4_2(fp5)

        x = x+fp5


        x = self.conv_down1(x)
        x = nn.functional.interpolate(x, size=(320, 320), mode='bilinear', align_corners=True)
        x = self.sigmoid(x)

        return x
class Bottleneck2(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck2, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        # for p in self.parameters():
        #     p.requires_grad = False

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        # out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# edge1-net
class ResNet3(nn.Module):
    def __init__(self):
        super(ResNet3, self).__init__()
        self.conv3_1 = BaseConv(512, 128, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_1 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv5_1 = BaseConv(128, 64, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv6_1 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv7_1 = BaseConv(64, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_out_1 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_out_2 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_out_3 = BaseConv(32, 1, 1, 1, activation=nn.Sigmoid(), use_bn=False)

        self.conv_ff1_1 = BaseConv(64,32,1,1,activation=nn.ReLU(),use_bn=True)
        self.inplanes = 32
        self.res_stack1 = self.make_stack(32, 4, stride=1)
        self.conv_ff1_2 = BaseConv(32,32,3,1,activation=nn.ReLU(),use_bn=True)
        self.conv_ff2_1 = BaseConv(64, 32, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.res_stack2 = self.make_stack(32, 4, stride=1)
        self.conv_ff2_2 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        # initialize parameters
        for p in self.parameters():
            p.requires_grad = False
        self.init_param()

    def init_param(self):
        # The following is initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.shape[0] * m.weight.shape[1]
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
    def make_stack(self, planes, blocks, stride=1):
        downsample = None
        layers = []

        if stride != 1 or self.inplanes != planes * Bottleneck3.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck3.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck3.expansion),
            )

        layers.append(Bottleneck3(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * Bottleneck3.expansion
        for i in range(1, blocks):
            layers.append(Bottleneck3(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, fp2, fp3):

        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        x = self.conv3_1(x)
        x = self.conv4_1(x)

        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        x = self.conv5_1(x)
        x = self.conv6_1(x)
        x = self.conv7_1(x)

        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        fp2 = torch.cat([x,fp2],1)
        fp2 = self.conv_ff1_1(fp2)
        fp2 = self.res_stack1(fp2)
        fp2 = self.conv_ff1_2(fp2)
        x = torch.add(fp2,x)

        x = self.conv_out_1(x)
        x = self.conv_out_2(x)

        fp3 = torch.cat([x,fp3],1)
        fp3 = self.conv_ff2_1(fp3)
        fp3 = self.res_stack2(fp3)
        fp3 = self.conv_ff2_2(fp3)
        x = torch.add(fp3,x)

        x = self.conv_out_3(x)

        return x
class Bottleneck3(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck3, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        #
        # for p in self.parameters():
        #     p.requires_grad = False

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        # out = self.relu(out)

        out += residual
        out = self.relu(out)

        return out

class SFAoatt(nn.Module):
    def __init__(self):
        super(SFAoatt, self).__init__()
        # lou kong
        self.vgg_1 = VGG_1()
        self.dmp_1 = BackEnd_1()
        self.conv_out_1_1 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_out_1_2 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_out_1_3 = BaseConv(32, 1, 1, 1, activation=nn.Sigmoid(), use_bn=False)
        # for p in self.parameters():
        #     p.requires_grad = False
        # self.vgg_2 = VGG_2()
        # self.dmp_2 = BackEnd_2()
        # self.conv_out_2 = BaseConv(32, 1, 1, 1, activation=nn.Sigmoid(), use_bn=False)

    def forward(self, input_1, input_2):
        # input1 = yuan tu; input2 = dbd;
        input1 = self.vgg_1(input_1, input_2)
        dmp_out_32_1, fp2, fp3, fp4 = self.dmp_1(*input1)
        dmp_out_32_1 = self.conv_out_1_1(dmp_out_32_1)
        dmp_out_32_1 = self.conv_out_1_2(dmp_out_32_1)
        fp5 = dmp_out_32_1
        dmp_out_1_1 = self.conv_out_1_3(dmp_out_32_1)


        return dmp_out_1_1, fp2, fp3, fp4, fp5
class VGG_1(nn.Module):
    def __init__(self):
        super(VGG_1, self).__init__()
        self.pool_1 = nn.MaxPool2d(2, 2)
        self.conv1_1_1 = BaseConv(4, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv1_2_1 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_1_1 = BaseConv(64, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_2_1 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_1_1 = BaseConv(128, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_2_1 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_3_1 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_1_1 = BaseConv(256, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_2_1 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_3_1 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_1_1 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_2_1 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_3_1 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        # for p in self.parameters():
        #     p.requires_grad = False

    def forward(self, input1, input2):
        input = torch.cat([input1, input2], 1)
        input = self.conv1_1_1(input)
        input = self.conv1_2_1(input)

        input = self.pool_1(input)
        input = self.conv2_1_1(input)
        conv2_2 = self.conv2_2_1(input)

        input = self.pool_1(conv2_2)
        input = self.conv3_1_1(input)
        input = self.conv3_2_1(input)
        conv3_3 = self.conv3_3_1(input)

        input = self.pool_1(conv3_3)
        input = self.conv4_1_1(input)
        input = self.conv4_2_1(input)
        conv4_3 = self.conv4_3_1(input)

        input = self.pool_1(conv4_3)
        input = self.conv5_1_1(input)
        input = self.conv5_2_1(input)
        conv5_3 = self.conv5_3_1(input)

        return conv2_2, conv3_3, conv4_3, conv5_3
class BackEnd_1(nn.Module):
    def __init__(self):
        super(BackEnd_1, self).__init__()
        # self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv1_1 = BaseConv(1024, 256, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_1 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv3_1 = BaseConv(512, 128, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_1 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv5_1 = BaseConv(256, 64, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv6_1 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv7_1 = BaseConv(64, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        # for p in self.parameters():
        #     p.requires_grad = False

    def forward(self, *input):
        conv2_2, conv3_3, conv4_3, conv5_3 = input

        input = nn.functional.interpolate(conv5_3, scale_factor=2, mode='bilinear', align_corners=True)
        input = torch.cat([input, conv4_3], 1)
        input = self.conv1_1(input)
        input = self.conv2_1(input)
        fp2 = input

        input = nn.functional.interpolate(input, scale_factor=2, mode='bilinear', align_corners=True)
        input = torch.cat([input, conv3_3], 1)
        input = self.conv3_1(input)
        input = self.conv4_1(input)
        fp3 = input

        input = nn.functional.interpolate(input, scale_factor=2, mode='bilinear', align_corners=True)
        input = torch.cat([input, conv2_2], 1)
        input = self.conv5_1(input)
        input = self.conv6_1(input)
        input = self.conv7_1(input)
        fp4 = input
        input = nn.functional.interpolate(input, scale_factor=2, mode='bilinear', align_corners=True)

        return input, fp2, fp3, fp4

class SFAoatt2(nn.Module):
    def __init__(self):
        super(SFAoatt2, self).__init__()
        # tian chong
        self.vgg_2 = VGG_2()
        self.dmp_2 = BackEnd_2()
        self.conv_out_2_1 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_out_2_2 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_out_2_3 = BaseConv(32, 1, 1, 1, activation=nn.Sigmoid(), use_bn=False)
        # for p in self.parameters():
        #     p.requires_grad = False

    def forward(self, input_1, input_2):
        # input1 = yuan tu; input2 = edge;
        input2 = self.vgg_2(input_1, input_2)
        dmp_out_32_2, fp2, fp3, fp4 = self.dmp_2(*input2)
        dmp_out_32_2 = self.conv_out_2_1(dmp_out_32_2)
        dmp_out_32_2 = self.conv_out_2_2(dmp_out_32_2)
        fp5 = dmp_out_32_2
        dmp_out_1_2 = self.conv_out_2_3(dmp_out_32_2)

        return dmp_out_1_2, fp2, fp3, fp4, fp5
class VGG_2(nn.Module):
    def __init__(self):
        super(VGG_2, self).__init__()
        self.pool_2 = nn.MaxPool2d(2, 2)
        self.conv1_1_2 = BaseConv(4, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
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
        # for p in self.parameters():
        #     p.requires_grad = False

    def forward(self, input1, input2):
        input = torch.cat([input1, input2], 1)
        input = self.conv1_1_2(input)
        input = self.conv1_2_2(input)
        input = self.pool_2(input)
        input = self.conv2_1_2(input)
        conv2_2 = self.conv2_2_2(input)

        input = self.pool_2(conv2_2)
        input = self.conv3_1_2(input)
        input = self.conv3_2_2(input)
        conv3_3 = self.conv3_3_2(input)

        input = self.pool_2(conv3_3)
        input = self.conv4_1_2(input)
        input = self.conv4_2_2(input)
        conv4_3 = self.conv4_3_2(input)

        input = self.pool_2(conv4_3)
        input = self.conv5_1_2(input)
        input = self.conv5_2_2(input)
        conv5_3 = self.conv5_3_2(input)

        return conv2_2, conv3_3, conv4_3, conv5_3
class BackEnd_2(nn.Module):
    def __init__(self):
        super(BackEnd_2, self).__init__()
        # self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv1_2 = BaseConv(1024, 256, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv3_2 = BaseConv(512, 128, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_2 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)

        self.conv5_2 = BaseConv(256, 64, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv6_2 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv7_2 = BaseConv(64, 32, 3, 1, activation=nn.ReLU(), use_bn=True)
        # for p in self.parameters():
        #     p.requires_grad = False

    def forward(self, *input):
        conv2_2, conv3_3, conv4_3, conv5_3 = input

        input = nn.functional.interpolate(conv5_3, scale_factor=2, mode='bilinear', align_corners=True)

        input = torch.cat([input, conv4_3], 1)
        input = self.conv1_2(input)
        input = self.conv2_2(input)
        fp2 = input

        input = nn.functional.interpolate(input, scale_factor=2, mode='bilinear', align_corners=True)
        input = torch.cat([input, conv3_3], 1)
        input = self.conv3_2(input)
        input = self.conv4_2(input)
        fp3 = input

        input = nn.functional.interpolate(input, scale_factor=2, mode='bilinear', align_corners=True)
        input = torch.cat([input, conv2_2], 1)
        input = self.conv5_2(input)
        input = self.conv6_2(input)
        input = self.conv7_2(input)
        fp4 = input

        input = nn.functional.interpolate(input, scale_factor=2, mode='bilinear', align_corners=True)

        return input, fp2, fp3, fp4

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