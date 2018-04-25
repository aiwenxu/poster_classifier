# from torchvision.models import resnet18 as ResNet
#
# resnet = ResNet(True)
#
# print(resnet)

# import pandas as pd
# from PIL import Image
# import os.path
#
#
# test = pd.read_csv("data/test_labels.csv", encoding="ISO-8859-1")
# validate = pd.read_csv("data/validate_labels.csv", encoding="ISO-8859-1")
# train = pd.read_csv("data/train_labels.csv", encoding="ISO-8859-1")
#
# print(len(test))
# print(len(validate))
# print(len(train))


from helper import clamp_probs
import torch

print(clamp_probs(torch.FloatTensor([0.5, 0.5, 0.2, 0.9, 0])))
