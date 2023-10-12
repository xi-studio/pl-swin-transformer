# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 21:37:17 2020
Unet + Resnet + Attention
@author: fzl
"""
# %%

import torch
import torch.nn as nn
from model import Attention_Assemble

# from convlstm import ConvBLSTM
from torch.nn.utils.parametrizations import spectral_norm


# %%
class ResBlock(torch.nn.Module):
    """
    func: ResNet模块，f = conv1(x) + conv1(x)
    Parameter
    ---------
    in_dim: int
        特征图的输入通道数
    out_dim: int
        输出通道数
    kernel_size: int
        卷积核尺寸，default 3
    stride: int
        卷积滑动步长，可选 1,2
        when 1: 保持原尺寸
        when 2: 尺寸减半

    """

    def __init__(self, in_dim, out_dim, stride=1, kernel_size=3, spectralnorm=False):
        super(ResBlock, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.stride = stride
        self.kernel_size = kernel_size

        self.padding = self.kernel_size // 2

        self.spectralnorm = spectralnorm

        if not self.spectralnorm:
            self.conv1 = torch.nn.Conv2d(
                self.in_dim,
                self.out_dim,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            )
            self.conv2 = torch.nn.Conv2d(
                self.out_dim,
                self.in_dim,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            )
        else:
            self.conv1 = spectral_norm(
                torch.nn.Conv2d(
                    self.in_dim,
                    self.out_dim,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                )
            )
            self.conv2 = spectral_norm(
                torch.nn.Conv2d(
                    self.out_dim,
                    self.in_dim,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                )
            )

        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual

        return out


# x = torch.randn((2,4,20,20))

# Res = ResBlock(4, 10)
# y = Res(x)

# print(y.size())
# print(y.max())


# %%
class ResAttenBlock(nn.Module):
    """
    func: Res + Atten, 残差 + 注意力模块
    Parameter
    ---------
    in_dim: int
        特征图的输入通道数
    attention_mode: str
        default: ECA
        options: ['SENet','CBAM','SKE','ECA','RecoNet','TripletAttention',
                  'SelfChannelAtten',NonLocal,'DANet',
                  'GCNet','CCNet','AxialChannel']
        分别对应不同的Attention机制

    """

    def __init__(self, in_dim, attention_mode="ECA"):
        super(ResAttenBlock, self).__init__()

        self.in_dim = in_dim
        self.out_dim = in_dim

        self.attention_mode = attention_mode
        assert self.attention_mode in [
            "SENet",
            "CBAM",
            "SKE",
            "ECA",
            "RecoNet",
            "TripletAttention",
            "SelfChannelAtten",
            "NonLocal",
            "DANet",
            "GCNet",
            "CCNet",
            "AxialChannel",
            "CBAtten_Res",
            "AttentionConv",
            None,
        ]

        if self.attention_mode == "SENet":
            self.atten = Attention_Assemble.SELayer(in_dim=self.out_dim)
        elif self.attention_mode == "CBAM":
            self.atten = Attention_Assemble.CBAtten(self.out_dim)
        elif self.attention_mode == "SKE":
            self.atten = Attention_Assemble.SKEConv(self.out_dim)
        elif self.attention_mode == "ECA":
            self.atten = Attention_Assemble.ECA(self.out_dim)
        elif self.attention_mode == "RecoNet":
            self.atten = Attention_Assemble.RecoNet(in_dim=self.out_dim, r=16)

        elif self.attention_mode == "TripletAttention":
            self.atten = Attention_Assemble.TripletAttention()

        elif self.attention_mode == "SelfChannelAtten":
            # print('SelfChannelAtten')
            self.atten = Attention_Assemble.Self_Attn_Channel(
                in_dim=self.out_dim, out_dim=self.out_dim
            )
        elif self.attention_mode == "NonLocal":
            self.atten = Attention_Assemble.NonLocalBlockND(
                in_channels=self.out_dim, dimension=2
            )
        elif self.attention_mode == "DANet":
            self.atten = Attention_Assemble.DANet(in_dim=self.out_dim)

        elif self.attention_mode == "GCNet":
            self.atten = Attention_Assemble.GCNet_Atten(in_dim=self.out_dim)

        elif self.attention_mode == "CCNet":
            self.atten = nn.ModuleList([])
            self.atten.append(Attention_Assemble.CC_module(in_dim=self.out_dim))
            self.atten.append(Attention_Assemble.CC_module(in_dim=self.out_dim))
        elif self.attention_mode == "CBAtten_Res":
            self.atten = Attention_Assemble.CBAtten_Res(self.out_dim)
        elif self.attention_mode == "AttentionConv":
            self.atten = Attention_Assemble.AttentionConv(
                in_channels=self.out_dim,
                out_channels=self.out_dim,
                kernel_size=1,
                groups=1,
                padding=0,
            )

    def forward(self, x):
        """
        Parameter
        --------
        x: 4D-Tensor  ----> (batch,channel,height,width)
        """
        x1 = x

        if self.attention_mode != None:
            if self.attention_mode == "CCNet":
                for ccnet in self.atten:
                    x1 = ccnet(x1)
            else:
                x1 = self.atten(x1)

                y = x1 + x
        else:
            y = x1

        return y


# device = torch.device('cuda:0')
# x = torch.randn((2,4,20,20)).to(device)

# resatten = ResAttenBlock(4,
#                           attention_mode = None).to(device)
# y = resatten(x)

# print(y.size())
# print(y.max())


# %%
class ConvBlock(torch.nn.Module):

    """
    func: 卷积模块，流程为: activation + conv + bn + dropout + res + atten
    Parameter
    ---------
    in_dim: int
        特征图的输入通道数
    out_dim: int
        输出通道数
    kernel_size: int
        卷积核尺寸，default 3
    stride: int
        卷积滑动步长，可选 1,2 . default 2
        when 1: 保持原尺寸
        when 2: 尺寸减半
    activation: bool
        default: True
        是否添加激活函数, 默认激活函数为LeakyRelu
    batch_norm: bool
        default: True. 是否添加BN层
    dropout: bool
        default: False. 是否添加Dropout层
    res: int
        default: 1, 添加几个Res层
    atten_mode: str or None
        when None, 则不加入Attention模块
        when str:  options: ['SENet','CBAM','SKE','ECA','RecoNet','TripletAttention',
                  'SelfChannelAtten',NonLocal,'DANet',
                  'GCNet','CCNet','AxialChannel']
                    分别对应不同的Attention机制
    Returns
    -------
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size=3,
        stride=2,
        activation=True,
        batch_norm=True,
        dropout=False,
        res=1,
        atten_mode="ECA",
        spectralnorm=False,
    ):
        super(ConvBlock, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation = activation
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.res = res
        self.atten_mode = atten_mode
        self.spectralnorm = spectralnorm

        self.padding = kernel_size // 2

        if not spectralnorm:
            self.conv = torch.nn.Conv2d(
                self.in_dim,
                self.out_dim,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            )
        else:
            self.conv = spectral_norm(
                torch.nn.Conv2d(
                    self.in_dim,
                    self.out_dim,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                )
            )

        if self.activation:
            self.lrelu = torch.nn.LeakyReLU(0.2, True)

        if self.batch_norm:
            self.bn = torch.nn.BatchNorm2d(self.out_dim, affine=True)

        if self.dropout:
            self.drop = torch.nn.Dropout(0.5)

        if self.res:
            BS = []
            for i in range(self.res):
                BS.append(
                    ResBlock(
                        self.out_dim,
                        self.out_dim,
                        stride=1,
                        kernel_size=3,
                        spectralnorm=self.spectralnorm,
                    )
                )
            self.Res = torch.nn.Sequential(*BS)

        if self.atten_mode:
            self.ResAtten = ResAttenBlock(
                in_dim=self.out_dim, attention_mode=self.atten_mode
            )

    def forward(self, x):
        """
        x: 4D-Tensor ---> [batch, in_dim, height, width]
        """
        if self.activation:
            x = self.lrelu(x)

        x = self.conv(x)

        if self.batch_norm:
            x = self.bn(x)
        if self.dropout:
            x = self.drop(x)

        if self.res:
            x = self.Res(x)

        if self.atten_mode:
            x = self.ResAtten(x)

        return x


# device = torch.device('cuda:0')
# device = torch.device('cpu')
# x = torch.randn((2,4,20,20)).to(device)

# Conv = ConvBlock(4, 10, stride = 1, res = 1,
#                  atten_mode = 'SelfChannelAtten').to(device)
# y = Conv(x)

# print(y.size())
# print(y.max())

# %%


class ASPP_ConvBlock(torch.nn.Module):
    """
    func: 空间金字塔池化-卷积模块
    Parameter
    ---------
    in_dim: int
        特征图的输入通道数
    out_dim: int
        输出通道数
    kernel_size: int
        卷积核尺寸，default 3;
    stride: int
        卷积滑动步长，可选 1,2 . default 2
        when 1: 保持原尺寸
        when 2: 尺寸减半
    activation: bool
        default: True
        是否添加激活函数, 默认激活函数为LeakyRelu
    batch_norm: bool
        default: True. 是否添加BN层
    dropout: bool
        default: False. 是否添加Dropout层
    res: int
        default: 1, 添加几个Res层
    atten_mode: str or None
        when None, 则不加入Attention模块
        when str:  options: ['SENet','CBAM','SKE','ECA','RecoNet','TripletAttention',
                  'SelfChannelAtten',NonLocal,'DANet',
                  'GCNet','CCNet','AxialChannel']
                    分别对应不同的Attention机制
    Returns
    -------

    """

    def __init__(
        self,
        in_dim,
        out_dim,
        stride=2,
        activation=True,
        batch_norm=True,
        dropout=False,
        res=1,
        atten_mode="SENet",
    ):
        super(ASPP_ConvBlock, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = 3
        self.stride = stride
        self.activation = activation
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.res = res
        self.atten_mode = atten_mode

        # 6个级联，分别对应：[conv(3,3),d=6,d=12,d=18,maxpool,meanpool],stride = 2, kernel_size = 3
        self.conv1 = torch.nn.Conv2d(
            self.in_dim,
            self.out_dim,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.kernel_size // 2,
        )
        self.bn1 = torch.nn.BatchNorm2d(self.out_dim, affine=True)

        self.conv2 = torch.nn.Conv2d(
            self.in_dim,
            self.out_dim,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dilation=6,
            padding=(6 + 6) // 2,
        )
        self.bn2 = torch.nn.BatchNorm2d(self.out_dim, affine=True)

        self.conv3 = torch.nn.Conv2d(
            self.in_dim,
            self.out_dim,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dilation=12,
            padding=(12 + 12) // 2,
        )
        self.bn3 = torch.nn.BatchNorm2d(self.out_dim, affine=True)

        self.conv4 = torch.nn.Conv2d(
            self.in_dim,
            self.out_dim,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dilation=18,
            padding=(18 + 18) // 2,
        )
        self.bn4 = torch.nn.BatchNorm2d(self.out_dim, affine=True)

        self.max_pool = torch.nn.MaxPool2d(
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.kernel_size // 2,
        )
        self.mean_pool = torch.nn.AvgPool2d(
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.kernel_size // 2,
        )

        self.weights = torch.nn.Parameter(torch.ones((6)), requires_grad=True)

        # 如果self.in_dim == self.out_dim
        if self.in_dim != self.out_dim:
            self.conv5 = torch.nn.Conv2d(self.in_dim, self.out_dim, kernel_size=1)
            self.bn5 = torch.nn.BatchNorm2d(self.out_dim, affine=True)
            self.conv6 = torch.nn.Conv2d(self.in_dim, self.out_dim, kernel_size=1)
            self.bn6 = torch.nn.BatchNorm2d(self.out_dim, affine=True)

        # 将6个分支进行channel整合，之后经过一个conv(1*1)，
        # 把通道维同 6*self.out_dim ---> self.out_dim
        self.conv_cat = torch.nn.Conv2d(6 * self.out_dim, self.out_dim, kernel_size=1)

        if self.activation:
            self.lrelu = torch.nn.LeakyReLU(0.2, True)

        if self.batch_norm:
            self.bn = torch.nn.BatchNorm2d(self.out_dim, affine=True)

        if self.dropout:
            self.drop = torch.nn.Dropout(0.5)

        if self.res:
            BS = []
            for i in range(self.res):
                BS.append(ResBlock(self.out_dim, self.out_dim))
            self.Res = torch.nn.Sequential(*BS)

        if self.atten_mode:
            self.ResAtten = ResAttenBlock(
                in_dim=self.out_dim, attention_mode=self.atten_mode
            )

    def forward(self, x):
        """
        x: 4D-Tensor ---> [batch, in_dim, height, width]
        """
        if self.activation:
            x = self.lrelu(x)

        # 给6个分支可学习权重
        #       print(x.device)
        self.weights = self.weights.to(x.device)

        x1 = self.bn1(self.conv1(x)) * self.weights[0]
        x2 = self.bn2(self.conv2(x)) * self.weights[1]
        x3 = self.bn3(self.conv3(x)) * self.weights[2]
        x4 = self.bn4(self.conv4(x)) * self.weights[3]
        x1 = self.bn1(self.conv1(x))
        x2 = self.bn2(self.conv2(x))
        x3 = self.bn3(self.conv3(x))
        x4 = self.bn4(self.conv4(x))
        x5 = self.max_pool(x)
        x6 = self.mean_pool(x)

        if self.in_dim != self.out_dim:
            x5 = self.bn5(self.conv5(x5))
            x6 = self.bn6(self.conv6(x6))

        x5 = x5 * self.weights[4]
        x6 = x6 * self.weights[5]

        # out.size = (batch,6*self.out_dim,h,w)
        out = torch.cat([x1, x2, x3, x4, x5, x6], axis=1)
        out = self.conv_cat(out)

        #         print('x1',x1.size())
        #         print('x2',x2.size())
        #         print('x3',x3.size())
        #         print('x4',x4.size())
        #         print('x5',x5.size())
        #         print('x6',x6.size())
        #         print(out.size())

        if self.batch_norm:
            out = self.bn(out)
        if self.dropout:
            out = self.drop(out)

        if self.res:
            out = self.Res(out)

        if self.atten_mode:
            out = self.ResAtten(out)

        return out


# device = torch.device('cuda:0')
# device = torch.device('cpu')
# x = torch.randn((2,4,128,128)).to(device)

# Conv = ASPP_ConvBlock(4, 10, stride = 1,
#                         res = 0,
#                         atten_mode = None
#                       ).to(device)

# y = Conv(x)
# # print(y.size())
# y.sum().backward()


class DeconvBlock(nn.Module):
    """
    func: 升级尺寸. 流程为: activation + upsample(or deconv) + bn + dropout + res + atten

    Parameter
    ---------
    in_dim: int
        特征图的输入通道数
    out_dim: int
        输出通道数
    kernel_size: int
        卷积核尺寸，default 3
    stride: int
        尺寸变换因子，可选 1,2
        when 1: 保持原尺寸
        when 2: 尺寸翻倍
    up_mode: str
        可选 ['upsample','devonv']
        when 'upsample': 通过nn.UpsamplingBilinear2d实现尺寸变换
        when 'deconv': 通过nn.ConvTranspose2d实现尺寸变换
    activation: bool
        default: True
        是否添加激活函数, 默认激活函数为LeakyRelu
    batch_norm: bool
        default: True. 是否添加BN层
    dropout: bool
        default: False. 是否添加Dropout层
    res: int
        default: 1, 添加几个Res层
    atten_mode: str or None
        when None, 则不加入Attention模块
        when str:  options: ['SENet','CBAM','SKE','ECA','RecoNet',
                  'SelfChannelAtten',NonLocal,'DANet',
                  'GCNet','CCNet','AxialChannel']
                    分别对应不同的Attention机制
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size=3,
        stride=2,
        up_mode="upsample",
        activation=True,
        batch_norm=True,
        dropout=False,
        res=1,
        atten_mode="ECA",
        spectralnorm=False,
    ):
        super(DeconvBlock, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.activation = activation
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.res = res
        self.atten_mode = atten_mode
        self.spectralnorm = spectralnorm

        self.up_mode = up_mode

        # 确保kernel_size为奇数
        assert kernel_size % 2 == 1
        assert up_mode in ["upsample", "deconv"]
        assert self.stride in [1, 2]

        if up_mode == "upsample":
            self.up = torch.nn.Upsample(
                scale_factor=self.stride, mode="bilinear", align_corners=True
            )

        else:
            # 当使用deconv进行尺寸升级时，
            # 当尺寸不变时，即stride = 1时，按照conv2d参数进行设置
            if self.stride == 1:
                self.up = nn.ConvTranspose2d(
                    self.in_dim,
                    self.in_dim,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=self.padding,
                )

            # 当尺寸升级一倍时，kernel_size =4,padding = 1, 才能保证尺寸翻倍
            elif self.stride == 2:
                self.up = nn.ConvTranspose2d(
                    self.in_dim, self.in_dim, kernel_size=4, stride=2, padding=1
                )

            self.bn0 = torch.nn.BatchNorm2d(self.in_dim, affine=True)
            self.relu0 = nn.ReLU(inplace=True)

        if not self.spectralnorm:
            self.conv1 = nn.Conv2d(
                self.in_dim,
                self.out_dim,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.padding,
            )

        else:
            self.conv1 = spectral_norm(
                nn.Conv2d(
                    self.in_dim,
                    self.out_dim,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=self.padding,
                )
            )

        if self.activation:
            self.relu = nn.ReLU(inplace=True)

        if self.batch_norm:
            self.bn = torch.nn.BatchNorm2d(self.out_dim, affine=True)

        if self.dropout:
            self.drop = torch.nn.Dropout(0.5)

        if self.res:
            BS = []
            for i in range(self.res):
                BS.append(
                    ResBlock(
                        self.out_dim,
                        self.out_dim,
                        stride=1,
                        kernel_size=3,
                        spectralnorm=self.spectralnorm,
                    )
                )
            self.Res = torch.nn.Sequential(*BS)

        if self.atten_mode:
            self.ResAtten = ResAttenBlock(
                in_dim=self.out_dim, attention_mode=self.atten_mode
            )

    def forward(self, x):
        """
        x: 4D-Tensor ---> [batch, in_dim, height, width]
        """

        if self.activation:
            x = self.relu(x)

        out = self.up(x)

        if self.up_mode == "deconv":
            out = self.bn0(out)
            out = self.relu0(out)

        out = self.conv1(out)

        if self.batch_norm:
            out = self.bn(out)

        if self.res:
            out = self.Res(out)

        if self.atten_mode:
            out = self.ResAtten(out)

        if self.dropout:
            out = self.drop(out)

        return out


# x = torch.randn((2,4,20,20))

# deconv = DeconvBlock(4, 10, stride = 2, up_mode = 'deconv', res = 1)
# y = deconv(x)

# print(y.size())
# print(y.max())


# %%
class UnetEncode(torch.nn.Module):
    """
    func: Unet的编码阶段
    Parameter
    ---------
    in_dim: int
        特征图的输入通道数
    base_dim: int
        基础通道数，default 32
        一般为32的整数倍
    layer: int
        垂直层数, default 4
    atten_mode: str or None
        when None, 则不加入Attention模块
        when str:  options: ['SENet','CBAM','SKE','ECA','RecoNet',
                  'SelfChannelAtten',NonLocal,'DANet',
                  'GCNet','CCNet','AxialChannel']
        分别对应不同的Attention机制
    ASPP: bool
        是否在encoder的每层的输出后加入ASPP(空间金字塔空洞卷积+池化-卷积模块)卷积模块
        default False.即不加入ASPP模块，when True.则加入1层ASPP模块
    Returns
    -------
    out: list
        每一层(共计layer层)最后的输出组成的list
    """

    def __init__(
        self,
        in_dim,
        base_dim=32,
        layer=4,
        atten_mode="ECA",
        ASPP=False,
        spectralnorm=False,
    ):
        super(UnetEncode, self).__init__()

        self.in_dim = in_dim
        self.base_dim = base_dim
        self.atten_mode = atten_mode

        self.layer = layer
        self.ASPP = ASPP
        self.spectralnorm = spectralnorm
        self.conv0 = ConvBlock(
            in_dim=self.in_dim,
            out_dim=self.base_dim,
            stride=1,
            activation=False,
            batch_norm=True,
            res=0,
            atten_mode=None,
            spectralnorm=self.spectralnorm,
        )

        self.conv_layers = nn.ModuleList()
        for i in range(self.layer):
            if i == 0:
                in_dim = self.base_dim
                out_dim = in_dim

            else:
                in_dim = out_dim
                out_dim = in_dim * 2

            self.conv = ConvBlock(
                in_dim=in_dim,
                out_dim=out_dim,
                stride=2,
                activation=True,
                batch_norm=True,
                res=1,
                atten_mode=self.atten_mode,
                spectralnorm=self.spectralnorm,
            )

            self.conv_layers.append(self.conv)

            ###############
            ## Add ASPP to the last layer's output
            ###############
            if self.ASPP:
                self.ASPP_all_layer = ASPP_ConvBlock(
                    in_dim=out_dim,
                    out_dim=out_dim,
                    stride=1,
                    activation=False,
                    batch_norm=True,
                    res=0,
                    dropout=False,
                    atten_mode=None,
                )

    def forward(self, x):
        """
        x: 4D-Tensor ---> [batch,in_dim, height, width]
        """
        out = []

        # size = (batch, base_dim, height, width)
        x = self.conv0(x)

        for conv in self.conv_layers:
            x = conv(x)
            # print(x.size())
            out.append(x)

        if self.ASPP:
            final_out = []
            final_out = out[:-1]
            final_out.append(self.ASPP_all_layer(out[-1]))

        return out if self.ASPP == False else final_out


# x = torch.randn((2,4,128,128))

# encode = UnetEncode(4, 32,layer = 4)
# y = encode(x)

# print(len(y))
# print([print(out.size()) for out in y])


# %%
class ComposeMoreEncode(torch.nn.Module):
    """
    func: 将多个Unet的编码阶段的结果进行综合处理
    Parameter
    ---------
    in_dim_list: list of int
        每个encode的输入通道数组成的list
    encode_num: int
        由几个encode组成
    base_dim: int
        基础通道数，default 32
        一般为32的整数倍
    layer: int
        垂直层数, default 4
    atten_mode: str or None
        when None, 则不加入Attention模块
        when str:  options: ['SENet','CBAM','SKE','ECA','RecoNet',
                  'SelfChannelAtten',NonLocal,'DANet',
                  'GCNet','CCNet','AxialChannel']
        分别对应不同的Attention机制
    compose_type: str
        可选['cat','add']
        when 'add': 则将多个encode阶段对应的layer输出直接相加
        when 'cat': 则先将多个encode阶段对应的layer的输出先通道维度拼接，之后
            通过conv + res + atten 降低到one_encode对应的layer输出的维度大小
        default: cat
    ASPP: bool
        是否在encoder的每层的输出后加入ASPP(空间金字塔空洞卷积+池化-卷积模块)卷积模块
        default False.即不加入ASPP模块，when True.则加入1层ASPP模块
    Returns
    -------
    out: list
        每一层(layer)最后的输出组成的list
    """

    def __init__(
        self,
        in_dim_list,
        encode_num,
        base_dim=32,
        layer=4,
        atten_mode="ECA",
        compose_type="cat",
        ASPP=False,
        spectralnorm=False,
    ):
        super(ComposeMoreEncode, self).__init__()

        self.in_dim_list = in_dim_list
        self.encode_num = encode_num
        self.base_dim = base_dim
        self.layer = layer
        self.atten_mode = atten_mode
        self.compose_type = compose_type
        self.ASPP = ASPP
        self.spectralnorm = spectralnorm

        assert isinstance(in_dim_list, list)
        assert len(in_dim_list) == encode_num
        assert compose_type in ["cat", "add"]

        self.encode_layers = nn.ModuleList()
        for in_dim in in_dim_list:
            self.encode = UnetEncode(
                in_dim=in_dim,
                base_dim=self.base_dim,
                layer=self.layer,
                atten_mode=self.atten_mode,
                ASPP=self.ASPP,
                spectralnorm=self.spectralnorm,
            )

            self.encode_layers.append(self.encode)

        if self.compose_type == "cat":
            self.conv_layers = torch.nn.ModuleList()
            for i in range(self.layer):
                in_dim = self.encode_num * self.base_dim * 2 ** (i)
                out_dim = self.base_dim * 2 ** (i)
                conv = ConvBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    stride=1,
                    activation=True,
                    batch_norm=True,
                    dropout=False,
                    res=2,
                    atten_mode=self.atten_mode,
                    spectralnorm=self.spectralnorm,
                )
                self.conv_layers.append(conv)

    def forward(self, X_list):
        """
        X_list: list or Tensor
            由encode_num个4D-Tensor组成的 list
        """

        # 确保输入的X_list的数量和对应通道数和in_dim_list一致
        assert len(X_list) == self.encode_num
        for i in range(self.encode_num):
            X_list[i].size()[1] == self.in_dim_list[i]

        # 获取每个输入x对应的encode的输出, 各个encode的输出维度应当一致
        all_encode_out = []
        for i in range(self.encode_num):
            x = X_list[i]
            out = self.encode_layers[i](x)
            all_encode_out.append(out)

        # 如果是add模型，则将所有encode的对应层数的encode_num个输出直接相加即可
        single_out = []
        if self.compose_type == "add":
            for i in range(self.layer):
                out = 0
                for j in range(self.encode_num):
                    out += all_encode_out[j][i]
                single_out.append(out)

        else:
            # 如果是cat模型，则将所有encode的对应层数的encode_num个输出在通道维进行拼接
            new_single_out = []
            for i in range(self.layer):
                out = []
                for j in range(self.encode_num):
                    out.append(all_encode_out[j][i])
                out = torch.cat(out, axis=1)
                new_single_out.append(out)

            # 拼接之后，使用卷积操作 + res + atten操作 将通道降低到单encode模型对应的输出的通道数
            for i, conv in enumerate(self.conv_layers):
                out = conv(new_single_out[i])
                single_out.append(out)

        return single_out


# x0 = torch.randn((2,4,128,128))
# x1 = torch.randn((2,8,128,128))

# more_encode = ComposeMoreEncode(in_dim_list = [4,8],
#                                 encode_num = 2,
#                                 base_dim = 32,
#                                 layer = 4,
#                                 compose_type = 'cat')
# y = more_encode([x0,x1])

# print(len(y))
# print([print(out.size()) for out in y])


# %%
class UnetDecode(torch.nn.Module):
    """
    func: Unet的解码阶段
    Parameter
    ---------
    in_dim_list: list or int
        多个输入的x的输入通道数组成的list
    out_dim: int
        解码最后的输出通道数
    base_dim: int
        基础通道数，default 32
        一般为32的整数倍
    layer: int
        垂直层数, default 4
    up_mode: str
        可选 ['upsample','devonv'].default 'upsample'.
        when 'upsample': 通过nn.UpsamplingBilinear2d实现尺寸变换
        when 'deconv': 通过nn.ConvTranspose2d实现尺寸变换
    atten_mode: str or None
        default 'ECA'
        when None, 则不加入Attention模块
        when str:  options: ['SENet','CBAM','SKE','ECA','RecoNet',
                  'SelfChannelAtten',NonLocal,'DANet',
                  'GCNet','CCNet','AxialChannel']
                    分别对应不同的Attention机制
    compose_type: str
        可选['cat','add']
        when 'add': 则将多个encode阶段对应的layer输出直接相加
        when 'cat': 则先将多个encode阶段对应的layer的输出先通道维度拼接，之后
            通过conv + res + atten 降低到one_encode对应的layer输出的维度大小
        default: cat
    ASPP: bool
        是否在encoder的每层的输出后加入ASPP(空间金字塔空洞卷积+池化-卷积模块)卷积模块
        default False.即不加入ASPP模块，when True.则加入1层ASPP模块
    Returns
    -------
    out: 4D-Tensor ---> [batch,out_dim, height, width]

    """

    def __init__(
        self,
        in_dim_list,
        out_dim,
        base_dim=32,
        layer=4,
        up_mode="upsample",
        atten_mode="ECA",
        compose_type="cat",
        ASPP=False,
        BCLSTM=False,
        spectralnorm=False,
    ):
        super(UnetDecode, self).__init__()

        self.in_dim_list = in_dim_list
        self.out_dim = out_dim
        self.base_dim = base_dim
        self.layer = layer
        self.up_mode = up_mode
        self.atten_mode = atten_mode
        self.compose_type = compose_type
        self.ASPP = ASPP
        self.BCLSTM = BCLSTM
        self.spectralnorm = spectralnorm

        if isinstance(self.in_dim_list, int):
            encode_in_dim = [self.in_dim_list]
        else:
            encode_in_dim = in_dim_list

        self.encode = ComposeMoreEncode(
            in_dim_list=encode_in_dim,
            encode_num=len(encode_in_dim),
            base_dim=self.base_dim,
            layer=self.layer,
            atten_mode=self.atten_mode,
            compose_type=self.compose_type,
            ASPP=self.ASPP,
            spectralnorm=self.spectralnorm,
        )

        self.bclstm_layers = nn.ModuleList()
        self.deconv_layers = nn.ModuleList()
        for i in range(self.layer):
            if i == 0:
                in_dim = self.base_dim * 2 ** (self.layer - 1)
                out_dim = in_dim // 2
                dropout = True
            else:
                in_dim = out_dim * 2
                out_dim = in_dim // 4
                dropout = False

                if self.BCLSTM:
                    self.bclstm_layer = ConvBLSTM(
                        in_channels=in_dim // 2,
                        hidden_channels=in_dim * 2,
                        kernel_size=(3, 3),
                        num_layers=1,
                        batch_first=True,
                    )
                    self.bclstm_layers.append(self.bclstm_layer)

            self.deconv = DeconvBlock(
                in_dim=in_dim,
                out_dim=out_dim,
                stride=2,
                up_mode=self.up_mode,
                activation=True,
                batch_norm=True,
                dropout=dropout,
                atten_mode=self.atten_mode,
                spectralnorm=self.spectralnorm,
            )

            self.deconv_layers.append(self.deconv)

            self.conv1 = ConvBlock(
                in_dim=out_dim,
                out_dim=self.out_dim,
                stride=1,
                activation=True,
                # batch_norm = False,
                res=0,
                atten_mode=None,
                spectralnorm=self.spectralnorm,
            )

    def forward(self, x_list):
        """
        x_list: list or Tensor
            when Tensor. 4D-Tensor or 5D-Tensor
                when 4D-Tensor, size = (batch, in_dim, height,width)
                when 5D-Tensor,则是由encode_num个4D-Tensor组成的 (batch,encode_num,in_dim,height,width)
                                此时的多个encoder的in_dim都是一致的
            when list. 由encode_num个4D-Tensor组成的 list
        """

        if isinstance(x_list, torch.Tensor) and len(x_list.size()) == 5:
            batch, nums, channel, height, width = x_list.size()
            # print(x_list.size(),nums, len(self.in_dim_list))
            assert nums == len(self.in_dim_list)
            x = []
            for i in range(nums):
                x.append(x_list[:, i, :, :])

            x_list = x

        if not isinstance(x_list, list):
            x_list = [x_list]

        # print(len(x_list))
        out = self.encode(x_list)

        for i, deconv in enumerate(self.deconv_layers):
            if i == 0:
                x = deconv(out[-1])
            elif self.BCLSTM:
                x_fwd = torch.stack([x, out[-(i + 1)]], dim=1)
                x_rev = torch.stack([out[-(i + 1)], x], dim=1)

                x = self.bclstm_layers[i - 1](x_fwd, x_rev)
                x = deconv(x)
                # print('after {} deconv [bclstm]'.format(i), x.size())
            else:
                x = torch.cat([x, out[-(i + 1)]], axis=1)
                x = deconv(x)

            # print(x.size())

        x = self.conv1(x)
        out = torch.nn.ReLU6()(x) / 6

        return out


# device = torch.device('cuda:0')

# x1 = torch.randn((4,4,256,256)).to(device)
# x2 = torch.randn((4,4,256,256)).to(device)
# x3 = torch.randn((4,4,256,256)).to(device)

# x = torch.stack([x1,x2,x3],dim = 1)

# decode = UnetDecode(in_dim_list = [4,4,4],
#                     out_dim = 2,
#                     base_dim = 32,
#                     layer = 3,
#                     atten_mode = 'TripletAttention',
#                     compose_type = 'cat').to(device)

# # y = decode([x1,x2,x3])
# y = decode(x)

# print(y.size())
# print(y.max())
# print(y.mean())
# print(y.min())
