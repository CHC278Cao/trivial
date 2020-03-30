
import os
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


STATE_DICT = {
    'block.0.0.weight': 'conv1.weight',
    "block.0.1.weight": "bn1.weight",
    "block.0.1.bias": "bn1.bias",
    "block.0.1.running_mean": "bn1.running_mean",
    "block.0.1.running_var": "bn1.running_var",
    'block.1.1.conv_bn1.conv.weight': 'layer1.0.conv1.weight',
    'block.1.1.conv_bn1.bn.weight': 'layer1.0.bn1.weight',
    'block.1.1.conv_bn1.bn.bias': 'layer1.0.bn1.bias',
    'block.1.1.conv_bn1.bn.running_mean': 'layer1.0.bn1.running_mean',
    'block.1.1.conv_bn1.bn.running_var': 'layer1.0.bn1.running_var',
    'block.1.1.conv_bn2.conv.weight': 'layer1.0.conv2.weight',
    'block.1.1.conv_bn2.bn.weight': 'layer1.0.bn2.weight',
    'block.1.1.conv_bn2.bn.bias': 'layer1.0.bn2.bias',
    'block.1.1.conv_bn2.bn.running_mean': 'layer1.0.bn2.running_mean',
    'block.1.1.conv_bn2.bn.running_var': 'layer1.0.bn2.running_var',
    'block.1.2.conv_bn1.conv.weight': 'layer1.1.conv1.weight',
    'block.1.2.conv_bn1.bn.weight': 'layer1.1.bn1.weight',
    'block.1.2.conv_bn1.bn.bias': 'layer1.1.bn1.bias',
    'block.1.2.conv_bn1.bn.running_mean': 'layer1.1.bn1.running_mean',
    'block.1.2.conv_bn1.bn.running_var': 'layer1.1.bn1.running_var',
    'block.1.2.conv_bn2.conv.weight': 'layer1.1.conv2.weight',
    'block.1.2.conv_bn2.bn.weight': 'layer1.1.bn2.weight',
    'block.1.2.conv_bn2.bn.bias': 'layer1.1.bn2.bias',
    'block.1.2.conv_bn2.bn.running_mean': 'layer1.1.bn2.running_mean',
    'block.1.2.conv_bn2.bn.running_var': 'layer1.1.bn2.running_var',
    'block.1.3.conv_bn1.conv.weight': 'layer1.2.conv1.weight',
    'block.1.3.conv_bn1.bn.weight': 'layer1.2.bn1.weight',
    'block.1.3.conv_bn1.bn.bias': 'layer1.2.bn1.bias',
    'block.1.3.conv_bn1.bn.running_mean': 'layer1.2.bn1.running_mean',
    'block.1.3.conv_bn1.bn.running_var': 'layer1.2.bn1.running_var',
    'block.1.3.conv_bn2.conv.weight': 'layer1.2.conv2.weight',
    'block.1.3.conv_bn2.bn.weight': 'layer1.2.bn2.weight',
    'block.1.3.conv_bn2.bn.bias': 'layer1.2.bn2.bias',
    'block.1.3.conv_bn2.bn.running_mean': 'layer1.2.bn2.running_mean',
    'block.1.3.conv_bn2.bn.running_var': 'layer1.2.bn2.running_var',
    'block.2.0.conv_bn1.conv.weight': 'layer2.0.conv1.weight',
    'block.2.0.conv_bn1.bn.weight': 'layer2.0.bn1.weight',
    'block.2.0.conv_bn1.bn.bias': 'layer2.0.bn1.bias',
    'block.2.0.conv_bn1.bn.running_mean': 'layer2.0.bn1.running_mean',
    'block.2.0.conv_bn1.bn.running_var': 'layer2.0.bn1.running_var',
    'block.2.0.conv_bn2.conv.weight': 'layer2.0.conv2.weight',
    'block.2.0.conv_bn2.bn.weight' : 'layer2.0.bn2.weight',
    'block.2.0.conv_bn2.bn.bias': 'layer2.0.bn2.bias',
    'block.2.0.conv_bn2.bn.running_mean': 'layer2.0.bn2.running_mean',
    'block.2.0.conv_bn2.bn.running_var': 'layer2.0.bn2.running_var',
    'block.2.0.shortcut.conv.weight': 'layer2.0.downsample.0.weight',
    'block.2.0.shortcut.bn.weight': 'layer2.0.downsample.1.weight',
    'block.2.0.shortcut.bn.bias': 'layer2.0.downsample.1.bias',
    'block.2.0.shortcut.bn.running_mean': 'layer2.0.downsample.1.running_mean',
    'block.2.0.shortcut.bn.running_var': 'layer2.0.downsample.1.running_var',
    'block.2.1.conv_bn1.conv.weight': 'layer2.1.conv1.weight',
    'block.2.1.conv_bn1.bn.weight': 'layer2.1.bn1.weight',
    'block.2.1.conv_bn1.bn.bias': 'layer2.1.bn1.bias',
    'block.2.1.conv_bn1.bn.running_mean': 'layer2.1.bn1.running_mean',
    'block.2.1.conv_bn1.bn.running_var': 'layer2.1.bn1.running_var',
    'block.2.1.conv_bn2.conv.weight': 'layer2.1.conv2.weight',
    'block.2.1.conv_bn2.bn.weight': 'layer2.1.bn2.weight',
    'block.2.1.conv_bn2.bn.bias': 'layer2.1.bn2.bias',
    'block.2.1.conv_bn2.bn.running_mean': 'layer2.1.bn2.running_mean',
    'block.2.1.conv_bn2.bn.running_var': 'layer2.1.bn2.running_var',
    'block.2.2.conv_bn1.conv.weight': 'layer2.2.conv1.weight',
    'block.2.2.conv_bn1.bn.weight': 'layer2.2.bn1.weight',
    'block.2.2.conv_bn1.bn.bias': 'layer2.2.bn1.bias',
    'block.2.2.conv_bn1.bn.running_mean': 'layer2.2.bn1.running_mean',
    'block.2.2.conv_bn1.bn.running_var': 'layer2.2.bn1.running_var',
    'block.2.2.conv_bn2.conv.weight': 'layer2.2.conv2.weight',
    'block.2.2.conv_bn2.bn.weight': 'layer2.2.bn2.weight',
    'block.2.2.conv_bn2.bn.bias': 'layer2.2.bn2.bias',
    'block.2.2.conv_bn2.bn.running_mean': 'layer2.2.bn2.running_mean',
    'block.2.2.conv_bn2.bn.running_var': 'layer2.2.bn2.running_var',
    'block.2.3.conv_bn1.conv.weight': 'layer2.3.conv1.weight',
    'block.2.3.conv_bn1.bn.weight': 'layer2.3.bn1.weight',
    'block.2.3.conv_bn1.bn.bias': 'layer2.3.bn1.bias',
    'block.2.3.conv_bn1.bn.running_mean': 'layer2.3.bn1.running_mean',
    'block.2.3.conv_bn1.bn.running_var': 'layer2.3.bn1.running_var',
    'block.2.3.conv_bn2.conv.weight': 'layer2.3.conv2.weight',
    'block.2.3.conv_bn2.bn.weight': 'layer2.3.bn2.weight',
    'block.2.3.conv_bn2.bn.bias': 'layer2.3.bn2.bias',
    'block.2.3.conv_bn2.bn.running_mean': 'layer2.3.bn2.running_mean',
    'block.2.3.conv_bn2.bn.running_var': 'layer2.3.bn2.running_var',
    'block.3.0.conv_bn1.conv.weight': 'layer3.0.conv1.weight',
    'block.3.0.conv_bn1.bn.weight': 'layer3.0.bn1.weight',
    'block.3.0.conv_bn1.bn.bias': 'layer3.0.bn1.bias',
    'block.3.0.conv_bn1.bn.running_mean': 'layer3.0.bn1.running_mean',
    'block.3.0.conv_bn1.bn.running_var': 'layer3.0.bn1.running_var',
    'block.3.0.conv_bn2.conv.weight': 'layer3.0.conv2.weight',
    'block.3.0.conv_bn2.bn.weight': 'layer3.0.bn2.weight',
    'block.3.0.conv_bn2.bn.bias': 'layer3.0.bn2.bias',
    'block.3.0.conv_bn2.bn.running_mean': 'layer3.0.bn2.running_mean',
    'block.3.0.conv_bn2.bn.running_var': 'layer3.0.bn2.running_var',
    'block.3.0.shortcut.conv.weight': 'layer3.0.downsample.0.weight',
    'block.3.0.shortcut.bn.weight': 'layer3.0.downsample.1.weight',
    'block.3.0.shortcut.bn.bias': 'layer3.0.downsample.1.bias',
    'block.3.0.shortcut.bn.running_mean': 'layer3.0.downsample.1.running_mean',
    'block.3.0.shortcut.bn.running_var': 'layer3.0.downsample.1.running_var',
    'block.3.1.conv_bn1.conv.weight': 'layer3.1.conv1.weight',
    'block.3.1.conv_bn1.bn.weight': 'layer3.1.bn1.weight',
    'block.3.1.conv_bn1.bn.bias': 'layer3.1.bn1.bias',
    'block.3.1.conv_bn1.bn.running_mean': 'layer3.1.bn1.running_mean',
    'block.3.1.conv_bn1.bn.running_var': 'layer3.1.bn1.running_var',
    'block.3.1.conv_bn2.conv.weight': 'layer3.1.conv2.weight',
    'block.3.1.conv_bn2.bn.weight': 'layer3.1.bn2.weight',
    'block.3.1.conv_bn2.bn.bias': 'layer3.1.bn2.bias',
    'block.3.1.conv_bn2.bn.running_mean': 'layer3.1.bn2.running_mean',
    'block.3.1.conv_bn2.bn.running_var': 'layer3.1.bn2.running_var',
    'block.3.2.conv_bn1.conv.weight': 'layer3.2.conv1.weight',
    'block.3.2.conv_bn1.bn.weight': 'layer3.2.bn1.weight',
    'block.3.2.conv_bn1.bn.bias': 'layer3.2.bn1.bias',
    'block.3.2.conv_bn1.bn.running_mean': 'layer3.2.bn1.running_mean',
    'block.3.2.conv_bn1.bn.running_var': 'layer3.2.bn1.running_var',
    'block.3.2.conv_bn2.conv.weight': 'layer3.2.conv2.weight',
    'block.3.2.conv_bn2.bn.weight': 'layer3.2.bn2.weight',
    'block.3.2.conv_bn2.bn.bias': 'layer3.2.bn2.bias',
    'block.3.2.conv_bn2.bn.running_mean': 'layer3.2.bn2.running_mean',
    'block.3.2.conv_bn2.bn.running_var': 'layer3.2.bn2.running_var',
    'block.3.3.conv_bn1.conv.weight': 'layer3.3.conv1.weight',
    'block.3.3.conv_bn1.bn.weight': 'layer3.3.bn1.weight',
    'block.3.3.conv_bn1.bn.bias': 'layer3.3.bn1.bias',
    'block.3.3.conv_bn1.bn.running_mean': 'layer3.3.bn1.running_mean',
    'block.3.3.conv_bn1.bn.running_var': 'layer3.3.bn1.running_var',
    'block.3.3.conv_bn2.conv.weight': 'layer3.3.conv2.weight',
    'block.3.3.conv_bn2.bn.weight': 'layer3.3.bn2.weight',
    'block.3.3.conv_bn2.bn.bias': 'layer3.3.bn2.bias',
    'block.3.3.conv_bn2.bn.running_mean': 'layer3.3.bn2.running_mean',
    'block.3.3.conv_bn2.bn.running_var': 'layer3.3.bn2.running_var',
    'block.3.4.conv_bn1.conv.weight': 'layer3.4.conv1.weight',
    'block.3.4.conv_bn1.bn.weight': 'layer3.4.bn1.weight',
    'block.3.4.conv_bn1.bn.bias': 'layer3.4.bn1.bias',
    'block.3.4.conv_bn1.bn.running_mean': 'layer3.4.bn1.running_mean',
    'block.3.4.conv_bn1.bn.running_var': 'layer3.4.bn1.running_var',
    'block.3.4.conv_bn2.conv.weight': 'layer3.4.conv2.weight',
    'block.3.4.conv_bn2.bn.weight': 'layer3.4.bn2.weight',
    'block.3.4.conv_bn2.bn.bias': 'layer3.4.bn2.bias',
    'block.3.4.conv_bn2.bn.running_mean': 'layer3.4.bn2.running_mean',
    'block.3.4.conv_bn2.bn.running_var': 'layer3.4.bn2.running_var',
    'block.3.5.conv_bn1.conv.weight': 'layer3.5.conv1.weight',
    'block.3.5.conv_bn1.bn.weight': 'layer3.5.bn1.weight',
    'block.3.5.conv_bn1.bn.bias': 'layer3.5.bn1.bias',
    'block.3.5.conv_bn1.bn.running_mean': 'layer3.5.bn1.running_mean',
    'block.3.5.conv_bn1.bn.running_var': 'layer3.5.bn1.running_var',
    'block.3.5.conv_bn2.conv.weight': 'layer3.5.conv2.weight',
    'block.3.5.conv_bn2.bn.weight': 'layer3.5.bn2.weight',
    'block.3.5.conv_bn2.bn.bias': 'layer3.5.bn2.bias',
    'block.3.5.conv_bn2.bn.running_mean': 'layer3.5.bn2.running_mean',
    'block.3.5.conv_bn2.bn.running_var': 'layer3.5.bn2.running_var',
    'block.4.0.conv_bn1.conv.weight': 'layer4.0.conv1.weight',
    'block.4.0.conv_bn1.bn.weight': 'layer4.0.bn1.weight',
    'block.4.0.conv_bn1.bn.bias': 'layer4.0.bn1.bias',
    'block.4.0.conv_bn1.bn.running_mean': 'layer4.0.bn1.running_mean',
    'block.4.0.conv_bn1.bn.running_var': 'layer4.0.bn1.running_var',
    'block.4.0.conv_bn2.conv.weight': 'layer4.0.conv2.weight',
    'block.4.0.conv_bn2.bn.weight': 'layer4.0.bn2.weight',
    'block.4.0.conv_bn2.bn.bias': 'layer4.0.bn2.bias',
    'block.4.0.conv_bn2.bn.running_mean': 'layer4.0.bn2.running_mean',
    'block.4.0.conv_bn2.bn.running_var': 'layer4.0.bn2.running_var',
    'block.4.0.shortcut.conv.weight': 'layer4.0.downsample.0.weight',
    'block.4.0.shortcut.bn.weight': 'layer4.0.downsample.1.weight',
    'block.4.0.shortcut.bn.bias': 'layer4.0.downsample.1.bias',
    'block.4.0.shortcut.bn.running_mean': 'layer4.0.downsample.1.running_mean',
    'block.4.0.shortcut.bn.running_var': 'layer4.0.downsample.1.running_var',
    'block.4.1.conv_bn1.conv.weight': 'layer4.1.conv1.weight',
    'block.4.1.conv_bn1.bn.weight': 'layer4.1.bn1.weight',
    'block.4.1.conv_bn1.bn.bias': 'layer4.1.bn1.bias',
    'block.4.1.conv_bn1.bn.running_mean': 'layer4.1.bn1.running_mean',
    'block.4.1.conv_bn1.bn.running_var': 'layer4.1.bn1.running_var',
    'block.4.1.conv_bn2.conv.weight': 'layer4.1.conv2.weight',
    'block.4.1.conv_bn2.bn.weight': 'layer4.1.bn2.weight',
    'block.4.1.conv_bn2.bn.bias': 'layer4.1.bn2.bias',
    'block.4.1.conv_bn2.bn.running_mean': 'layer4.1.bn2.running_mean',
    'block.4.1.conv_bn2.bn.running_var': 'layer4.1.bn2.running_var',
    'block.4.2.conv_bn1.conv.weight': 'layer4.2.conv1.weight',
    'block.4.2.conv_bn1.bn.weight': 'layer4.2.bn1.weight',
    'block.4.2.conv_bn1.bn.bias': 'layer4.2.bn1.bias',
    'block.4.2.conv_bn1.bn.running_mean': 'layer4.2.bn1.running_mean',
    'block.4.2.conv_bn1.bn.running_var': 'layer4.2.bn1.running_var',
    'block.4.2.conv_bn2.conv.weight':  'layer4.2.conv2.weight',
    'block.4.2.conv_bn2.bn.weight': 'layer4.2.bn2.weight',
    'block.4.2.conv_bn2.bn.bias': 'layer4.2.bn2.bias',
    'block.4.2.conv_bn2.bn.running_mean': 'layer4.2.bn2.running_mean',
    'block.4.2.conv_bn2.bn.running_var': 'layer4.2.bn2.running_var',
    'block.5.weight': 'fc.weight',
    'block.5.bias': 'fc.bias',
}


def load_pretained_weights(model, pretained_file, state_dict = STATE_DICT, skip = [], first_layer = None):
    """
        load pretrained weights with given matched state dict
    :param model: net model
    :param pretained_file: type: file, pretrained weights file
    :param state_dict: type: dict, state parameters dict
    :param skip: type: list, layers to be skipped
    :param first_layer: type: list, modify the first layer's pretrained weights for inputs which has one channel
    :return:
        None
    """
    assert os.path.exists(pretained_file), "pretrained file doesn't exist"
    print("loading pretrained weights to model")

    pretained_state_dict = torch.load(pretained_file, map_location=lambda storage, loc: storage)
    model_state_dict = model.state_dict()

    # pdb.set_trace()
    if first_layer is not None:
        for x in first_layer:
            weights = pretained_state_dict[state_dict[x]].clone()
            model_state_dict[x] = torch.nn.Parameter(torch.div(weights.sum(dim=1).unsqueeze(dim=1), 3))
            state_dict.pop(x)

    for model_key, pretrain_keys in state_dict.items():
        if model_key in skip:
            continue

        print("{} - {}".format(model_key, model_state_dict[model_key].shape))
        print("{} - {}".format(pretrain_keys, pretained_state_dict[pretrain_keys].shape))
        model_state_dict[model_key] = pretained_state_dict[pretrain_keys]

    pdb.set_trace()
    model.load_state_dict(model_state_dict)


class Resnet34_classification(nn.Module):
    def __init__(self, num_classes, dropout):
        super(Resnet34_classification, self).__init__()
        e = ResNet34()

        self.block = nn.ModuleList([
            e.block0,
            e.block1,
            e.block2,
            e.block3,
            e.block4,
        ])
        e = None
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=32, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        for conv in self.block:
            x = conv(x)

        x = self.dropout(x)
        x = self.pool(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return out


class ResNet34(nn.Module):
    def __init__(self, num_classes = 1000):
        super(ResNet34, self).__init__()

        self.block0 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, padding=3, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.block1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2),
            BasicBlock(in_channel=64, hidden_channel=64, out_channel=64, stride=1, is_shortcut=False),
            *[BasicBlock(in_channel=64, hidden_channel=64, out_channel=64, stride=1, is_shortcut=False)
              for _ in range(0, 2)],
        )
        self.block2 = nn.Sequential(
            BasicBlock(in_channel=64, hidden_channel=128, out_channel=128, stride=2, is_shortcut=True),
            *[BasicBlock(in_channel=128, hidden_channel=128, out_channel=128, stride=1, is_shortcut=False)
              for _ in range(0, 3)],
        )
        self.block3 = nn.Sequential(
            BasicBlock(in_channel=128, hidden_channel=256, out_channel=256, stride=2, is_shortcut=True),
            *[BasicBlock(in_channel=256, hidden_channel=256, out_channel=256, is_shortcut=False)
              for _ in range(0, 5)],
        )
        self.block4 = nn.Sequential(
            BasicBlock(in_channel=256, hidden_channel=512, out_channel=512, stride=2, is_shortcut=True),
            *[BasicBlock(in_channel=512, hidden_channel=512, out_channel=512, is_shortcut=False)
              for _ in range(0, 2)],
        )
        self.linear = nn.Linear(in_features=512, out_features=num_classes)

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x).reshape(x.size(0), -1)
        out = self.linear(x)

        return out


class BasicBlock(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel, stride = 1, is_shortcut = False):
        super(BasicBlock, self).__init__()

        self.is_shortcut = is_shortcut
        self.conv_bn1 = ConvBn2d(in_channel=in_channel, out_channel=hidden_channel,
                                     kernel_size=3, padding=1, stride=stride)
        self.conv_bn2 = ConvBn2d(in_channel=hidden_channel, out_channel=out_channel,
                                     kernel_size=3, padding=1, stride=1)
        if is_shortcut:
            self.shortcut = ConvBn2d(in_channel=in_channel, out_channel=out_channel,
                                     kernel_size=1, padding=0, stride=stride)
        self.relu = nn.ReLU()


    def forward(self, x):
        out = self.relu(self.conv_bn1(x))
        out = self.conv_bn2(out)
        if self.is_shortcut:
            x = self.shortcut(x)
        out = out + x
        out = self.relu(out)

        return out


class ConvBn2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size = 3, padding = 1, stride = 1):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                              kernel_size=kernel_size, padding=padding,
                              stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_channel, eps=1e-5)

    def forward(self, x):
        x = self.bn(self.conv(x))

        return x





