import torch
import torch.nn as nn


class ResBlock(torch.nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(ResBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.conv1.weight.data.normal_(0.0, 0.04)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.conv2.weight.data.normal_(0.0, 0.04)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        return out


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, lrelu=True):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(input_nc, ngf, 4, 2, 1)
        # self.res1 = ResBlock(ngf,ngf)
        self.conv2 = nn.Conv2d(ngf, ngf * 2, 4, 2, 1)
        # self.res2 = ResBlock(ngf * 2,ngf * 2)
        self.conv3 = nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1)
        # self.res3 = ResBlock(ngf * 4,ngf * 4)
        self.conv4 = nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1)
        # self.res4 = ResBlock(ngf * 8,ngf * 8)
        self.conv5 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
        # self.res5 = ResBlock(ngf * 8,ngf * 8)
        self.conv6 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
        # self.res6 = ResBlock(ngf * 8,ngf * 8)
        self.conv7 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
        # self.res7 = ResBlock(ngf * 8,ngf * 8)
        self.conv8 = nn.Conv2d(ngf * 8, ngf * 8, 4, 2, 1)
        # self.res8 = ResBlock(ngf * 8,ngf * 8)

        self.dconv1 = nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.dconv2 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1)
        self.dconv3 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1)
        self.dconv4 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1)
        self.dconv5 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, 4, 2, 1)
        self.dconv6 = nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, 4, 2, 1)
        self.dconv7 = nn.ConvTranspose2d(ngf * 2 * 2, ngf, 4, 2, 1)
        self.dconv8 = nn.ConvTranspose2d(ngf * 2, output_nc, 4, 2, 1)
        if lrelu:
            self.leaky_relu = nn.LeakyReLU(0.2, True)
        else:
            self.leaky_relu = nn.ReLU(True)

        self.relu = nn.ReLU(True)

        self.dropout = nn.Dropout(0.5)

        self.tanh = nn.Tanh()

    def forward(self, input):
        # Encoder
        # Convolution layers:
        # input is (nc) x 256 x 256
        e1 = self.conv1(input)
        # e1 = self.res1(e1)
        # state size is (ngf) x 128 x 128
        e2 = self.conv2(self.leaky_relu(e1))
        # e2 = self.res2(e2)
        # state size is (ngf x 2) x 64 x 64
        e3 = self.conv3(self.leaky_relu(e2))
        # e3 = self.res3(e3)
        # state size is (ngf x 4) x 32 x 32
        e4 = self.conv4(self.leaky_relu(e3))
        # e4 = self.res4(e4)
        # state size is (ngf x 8) x 16 x 16
        e5 = self.conv5(self.leaky_relu(e4))
        # e5 = self.res5(e5)
        # state size is (ngf x 8) x 8 x 8
        e6 = self.conv6(self.leaky_relu(e5))
        # e6 = self.res6(e6)
        # state size is (ngf x 8) x 4 x 4
        e7 = self.conv7(self.leaky_relu(e6))
        # e7 = self.res7(e7)
        # state size is (ngf x 8) x 2 x 2
        e8 = self.conv8(self.leaky_relu(e7))
        # e8 = self.res8(e8)
        # Decoder
        # Deconvolution layers:
        # state size is (ngf x 8) x 1 x 1
        d1_ = self.dconv1(self.relu(e8))
        # state size is (ngf x 8) x 2 x 2
        d1 = torch.cat((d1_, e7), 1)
        d2_ = self.dconv2(self.relu(d1))
        # state size is (ngf x 8) x 4 x 4
        d2 = torch.cat((d2_, e6), 1)
        d3_ = self.dconv3(self.relu(d2))
        # state size is (ngf x 8) x 8 x 8
        d3 = torch.cat((d3_, e5), 1)
        d4_ = self.dconv4(self.relu(d3))
        # state size is (ngf x 8) x 16 x 16
        d4 = torch.cat((d4_, e4), 1)
        d5_ = self.dconv5(self.relu(d4))
        # state size is (ngf x 4) x 32 x 32
        d5 = torch.cat((d5_, e3), 1)
        d6_ = self.dconv6(self.relu(d5))
        # state size is (ngf x 2) x 64 x 64
        d6 = torch.cat((d6_, e2), 1)
        d7_ = self.dconv7(self.relu(d6))
        # state size is (ngf) x 128 x 128
        d7 = torch.cat((d7_, e1), 1)
        d8 = self.dconv8(self.relu(d7))
        # state size is (nc) x 256 x 256
        output = self.tanh(d8)
        return output


class Discriminator(nn.Module):
    def __init__(self, input_nc, output_nc, ndf):
        super(Discriminator, self).__init__()
        # 256 x 256
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_nc + output_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 128 x 128
        self.layer2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 64 x 64
        self.layer3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 32 x 32
        self.layer4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 16 x 16
        self.layer5 = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=3, stride=1, padding=1), nn.Sigmoid()
        )
        # 16 x 16

    #
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out
