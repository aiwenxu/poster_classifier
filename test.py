# Testing on single-label data.

import torch
import torch.nn as nn
from torchvision.transforms import Scale, CenterCrop, ToTensor, Normalize, Compose
from torchvision.models import resnet18 as ResNet
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PosterDataset import PosterDataset
from helper import quick_print

def main():

    NET_PARAMS_DIR = "results/2/net_params"

    TEST_LABELS_DIR = "data/singlelabel/test_labels.csv"
    DATA_DIR = "data/posters"

    NUM_LABELS = 25

    data_transforms = Compose([Scale(268), CenterCrop(224), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    test_dataset = PosterDataset(csv_file=TEST_LABELS_DIR, root_dir=DATA_DIR, transform=data_transforms)

    test_data_size = len(test_dataset)

    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=4)

    use_gpu = torch.cuda.is_available()

    poster_net = ResNet(pretrained=True)
    for param in poster_net.parameters():
        param.requires_grad = False
    num_ftrs = poster_net.fc.in_features
    poster_net.fc = nn.Linear(num_ftrs, NUM_LABELS)
    poster_net.load_state_dict(torch.load(NET_PARAMS_DIR))

    if use_gpu:
        poster_net = poster_net.cuda()

    corrects = 0

    for data in test_loader:

        quick_print("...")

        img, label, title, imdb_id = data

        if use_gpu:
            img = Variable(img.cuda())
            label = Variable(label.cuda())
        else:
            img, label = Variable(img), Variable(label)

        scores = poster_net(img)
        _, preds = torch.max(scores, 1)

        corrects += torch.sum(preds.data == label.data)

    final_acc = corrects / test_data_size

    quick_print("final accuracy: {}".format(final_acc))


if __name__ == '__main__':
    main()
