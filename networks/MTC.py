import torch
from torch import nn
import torch.nn.functional as F
import torchinfo
import timm


# Basic residual block.
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y += self.shortcut(x)
        preact = y
        y = F.relu(y)
        if self.is_last:
            return y, preact
        else:
            return y


# Plain 2x 3x3 conv block (no residual connection).
class PlainBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(PlainBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y = F.relu(y)  # 去除残差，仅保留非线性激活
        return (y, y) if self.is_last else y

# ResNet (4 stages)
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes =64
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Kaiming init for convs, unit gamma / zero beta for norms.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-init residual’s last BN (helps training very deep nets; noop for PlainBlock)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0.0)

    # Build one ResNet stage with `num_blocks` blocks.
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=100):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


# ResNet up to layer3 (3 stages)
class ResNet3Block(ResNet):
    def __init__(self, block, num_blocks, in_channel=3):
        # Build as 4-stage and drop layer4.
        super(ResNet3Block, self).__init__(block, num_blocks + [0], in_channel=in_channel)
        del self.layer4  # remove the 4th stage

    def forward(self, x, layer=100):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


# ResNet with an extra 5th stage.
class ResNet5Block(ResNet):
    def __init__(self, block, num_blocks, in_channel=3):
        super(ResNet5Block, self).__init__(block, num_blocks + [2], in_channel=in_channel)
        self.layer5 = self._make_layer(block, 1024, 2, stride=2)  # extra stage

    def forward(self, x, layer=100):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def plaincnn18(**kwargs):
    return ResNet(PlainBlock, [2, 2, 2, 2], **kwargs)

def resnet18_3block(**kwargs):
    return ResNet3Block(BasicBlock, [2, 2, 2], **kwargs)

def resnet18_5block(**kwargs):
    return ResNet5Block(BasicBlock, [2, 2, 2, 2], **kwargs)


# ViT-Ti/16 with classifier removed.
def vit_tiny(pretrained=True):
    model = timm.create_model('vit_tiny_patch16_224', pretrained=pretrained)
    model.reset_classifier(0)
    return model

# ConvNeXt-Tiny with classifier removed.
def convnext_tiny(pretrained=True):
    model = timm.create_model('convnext_tiny', pretrained=pretrained)
    model.reset_classifier(0)
    return model


# name -> (constructor, feature_dim)
model_dict ={
    'resnet18': [resnet18, 512],
    'plaincnn18': [plaincnn18, 512],
    'resnet18_3block': [resnet18_3block, 256],      # layer3 out channels = 256
    'resnet18_5block': [resnet18_5block, 1024],     # extra stage -> 1024
    'vit_tiny': [vit_tiny, 192],
    'convnext_tiny': [convnext_tiny, 768],
}


# Model with encoder and projection head.
class SupConResNet(nn.Module):
    def __init__(self, encoder='resnet18', head='mlp', feat_dim=128):
        super(SupConResNet, self).__init__()
        model_fun, dim_in = model_dict[encoder]
        self.encoder = model_fun()
        if head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat


# Test Model and display layer-wise output shapes.
if __name__ == '__main__':
    model = SupConResNet(encoder='resnet18')
    input_size = (128, 3, 224, 224)  # (batch, C, H, W)
    torchinfo.summary(model, input_size=input_size)

