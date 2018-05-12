# The ResNet that can be used to achieve multi-label classification.

import torch.nn as nn
from torchvision.models import resnet18 as ResNet

class MultilabelFC(nn.Module):

    def __init__(self, input_dim, output_dim):

        super().__init__()

        self.linear = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        out = self.linear(x)
        out = self.sigmoid(out)

        return out

def get_customized_resnet(output_dim):

    customized_resnet = ResNet(pretrained=True)
    for param in customized_resnet.parameters():
        param.requires_grad = False

    num_ftrs = customized_resnet.fc.in_features
    customized_resnet.fc = MultilabelFC(num_ftrs, output_dim)

    return customized_resnet