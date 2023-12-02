from torchvision.models import ResNet
from torchvision.models.resnet import Bottleneck, BasicBlock

from .cbam import CBAM
import torch.utils.model_zoo as model_zoo
import logging
import torch
import torch.nn as nn

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class Backbone(ResNet):
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self, depth=101, cutmix=None):
        self.block, self.layers = self.arch_settings[depth]
        self.depth = depth
        super(Backbone, self).__init__(self.block, self.layers)
        self.init_weights(pretrained=True, cutmix=cutmix)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def init_weights(self, pretrained=True, cutmix=None):
        if cutmix is not None:
            print("backbone params inited by CutMix pretrained model")
            state_dict = torch.load(cutmix)
        elif pretrained:
            print("backbone params inited by Pytorch official model")
            model_url = model_urls["resnet{}".format(self.depth)]
            state_dict = model_zoo.load_url(model_url)

        model_dict = self.state_dict()
        try:
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            self.load_state_dict(pretrained_dict)
        except:
            logger = logging.getLogger()
            logger.info(
                "the keys in pretrained model is not equal to the keys in the ResNet you choose, trying to fix...")
            state_dict = self._keysFix(model_dict, state_dict)
            self.load_state_dict(state_dict)

        self.fc = nn.Sequential()


class ResNet_CSRA(nn.Module):
    def __init__(self, num_heads, lam, num_classes, num_Symptom_classes, depth=101, input_dim=2048, cutmix=None):
        super(ResNet_CSRA, self).__init__()

        self.backbone1 = Backbone(depth, cutmix)
        self.backbone2 = Backbone(depth, cutmix)
        self.branch_bam3 = CBAM(input_dim)
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(input_dim, num_classes)
        self.fc2 = nn.Linear(input_dim, num_Symptom_classes)

    def forward_train(self, x):

        x1 = self.backbone1(x)
        x2 = self.backbone2(x)
        x3 = self.branch_bam3(x1)
        x4 = x2 + x3

        x5 = self.avgpool1(x1)
        x5 = torch.flatten(x5, 1)
        logit1 = self.fc1(x5)
        x4 = self.avgpool2(x4)
        x4 = torch.flatten(x4, 1)
        logit2 = self.fc2(x4)
        logit = torch.hstack((logit1, logit2))
        return logit1, logit2, logit, x1, x2

    def forward_test(self, x):
        x1 = self.backbone1(x)
        x2 = self.backbone2(x)
        x3 = self.branch_bam3(x1)
        x4 = x2 + x3
        temp = x4
        x5 = self.avgpool1(x1)
        x5 = torch.flatten(x5, 1)
        logit1 = self.fc1(x5)
        x4 = self.avgpool2(x4)
        x4 = torch.flatten(x4, 1)
        logit2 = self.fc2(x4)

        return logit2, logit1, temp

    def forward(self, x, flag):
        if flag == True:
            return self.forward_train(x)
        else:
            return self.forward_test(x)
