
import torch
import torch.nn as nn
import torch.nn.functional as F

from res34Net import ResNet34


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Decode(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Decode, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel // 2),
            nn.ReLU(),

            nn.Conv2d(out_channel // 2, out_channel // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel//2),
            nn.ReLU(),

            nn.Conv2d(out_channel // 2, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class res34Unet(nn.Module):
    def __init__(self, num_classes):
        super(res34Unet, self).__init__()

        e = ResNet34()
        self.block = nn.ModuleList([
            e.block0,
            e.block1,
            e.block2,
            e.block3,
            e.block4
        ])
        e = None

        self.conv1 = nn.Conv2d(in_channels=512, out_channels=32, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.decode1 = Decode(in_channel=512, out_channel=512)
        self.decode2 = Decode(in_channel=256 + 512, out_channel=256)
        self.decode3 = Decode(in_channel=128 + 256, out_channel=256)
        self.decode4 = Decode(in_channel=64 + 256, out_channel=128)
        self.decode5 = Decode(in_channel=64 + 128, out_channel=128)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.logit = nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=1)

    def forward(self, inputs):
        height, width = inputs.shape[2:]

        downSample = []
        for i in range(len(self.block)):
            inputs = self.block[i](inputs)
            downSample.append(inputs)

        cls_feature = self.pool(downSample[-1])
        cls_out = self.conv1(cls_feature)
        cls_out = self.conv2(cls_out)

        out = self.decode1(downSample[-1])
        out = self.upsample(out)
        out = self.decode2(torch.cat([out, downSample[-2]], dim=1))
        out = self.upsample(out)
        out = self.decode3(torch.cat([out, downSample[-3]], dim=1))
        out = self.upsample(out)
        out = self.decode4(torch.cat([out, downSample[-4]], dim=1))
        out = self.upsample(out)
        out = self.decode5(torch.cat([out, downSample[-5]], dim=1))
        out = self.logit(out)
        seg_out = F.interpolate(out, size=(height, width), mode='bilinear', align_corners=False)

        return cls_out, seg_out









