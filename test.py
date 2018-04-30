#TODO

import torch
from torchvision.transforms import Scale, CenterCrop, ToTensor, Normalize, Compose
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PosterDataset import PosterDataset
from CustomizedResNet import get_customized_resnet
from helper import clamp_probs

def main():

    NET_PARAMS_DIR = "net_params"

    TEST_LABELS_DIR = "data/test_labels.csv"
    DATA_DIR = "data/posters"

    NUM_LABELS = 28

    data_transforms = Compose([Scale(268), CenterCrop(224), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    test_dataset = PosterDataset(csv_file=TEST_LABELS_DIR, root_dir=DATA_DIR, transform=data_transforms)

    test_loader = DataLoader(test_dataset)

    use_gpu = torch.cuda.is_available()

    poster_net = get_customized_resnet(NUM_LABELS)
    poster_net.load_state_dict(torch.load(NET_PARAMS_DIR))

    if use_gpu:
        poster_net = poster_net.cuda()

    for data in test_loader:

        img, label, title, imdb_id = data

        if use_gpu:
            img = Variable(img.cuda())
            label = Variable(label.cuda())
        else:
            img, label = Variable(img), Variable(label)

        prediction = poster_net(img)
        pred_label = clamp_probs(prediction.data[0])
        print(pred_label)
        print(label.data[0])
        input()


if __name__ == '__main__':
    main()
