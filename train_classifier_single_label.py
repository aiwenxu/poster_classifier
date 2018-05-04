import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Scale, RandomCrop, CenterCrop, ToTensor, Normalize, Compose
from torchvision.models import resnet18 as ResNet
import time
import copy
import os
from PosterDataset import PosterDataset
from helper import quick_print, clamp_probs, pickle_stat

def main():

    # Use HPC to train with different hyperparameters.
    learning_rates = [0.0005, 0.0001, 0.00005]
    learning_rate_decays = [0.8, 0.9]
    parameters = [(lr, lrd) for lr in learning_rates for lrd in learning_rate_decays]
    SLURM_TASK_ID = int(os.environ['SLURM_ARRAY_TASK_ID'])
    lr = parameters[SLURM_TASK_ID][0]
    lrd = parameters[SLURM_TASK_ID][1]
    quick_print("SLURM TASK {} has learning rate {}, decay ratio {}".format(SLURM_TASK_ID, lr, lrd))

    # Data.

    TRAIN_LABELS_DIR = "data/singlelabel/train_labels.csv"
    VALIDATE_LABELS_DIR = "data/singlelabel/validate_labels.csv"
    DATA_DIR = "data/posters"

    NUM_LABELS = 25

    data_transforms = {"train": Compose([Scale(268), RandomCrop(224), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
                       "val": Compose([Scale(268), CenterCrop(224), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])}

    train_dataset = PosterDataset(csv_file=TRAIN_LABELS_DIR, root_dir=DATA_DIR, transform=data_transforms["train"])
    val_dataset = PosterDataset(csv_file=VALIDATE_LABELS_DIR, root_dir=DATA_DIR, transform=data_transforms["val"])
    poster_datasets = {"train": train_dataset, "val": val_dataset}

    loaders = {x: DataLoader(poster_datasets[x], batch_size=128, shuffle=True, num_workers=4) for x in ["train", "val"]}

    dataset_sizes = {x: len(poster_datasets[x]) for x in ["train", "val"]}

    # Training details.

    use_gpu = torch.cuda.is_available()
    num_epochs = 200
    learning_rate = lr
    decay_ratio = lrd
    decay_epoch = 7

    # Create folders to save the training stats and results.

    folder_name = str(SLURM_TASK_ID)
    os.mkdir(folder_name)

    # Build the model.

    poster_net = ResNet(pretrained=True)
    for param in poster_net.parameters():
        param.requires_grad = False
    num_ftrs = poster_net.fc.in_features
    poster_net.fc = nn.Linear(num_ftrs, NUM_LABELS)

    if use_gpu:
        poster_net = poster_net.cuda()

    # Define the loss function, the optimizer and the learning rate decay.

    criterion = nn.CrossEntropyLoss()

    net_optimizer = optim.Adam(poster_net.fc.parameters(), lr=learning_rate)

    exp_lr_scheduler = lr_scheduler.StepLR(net_optimizer, step_size=decay_epoch, gamma=decay_ratio)

    # Start training.

    since = time.time()

    # Set up the storage for the losses and accuracies over training.
    val_losses = []
    val_accs = []

    best_model_wts = copy.deepcopy(poster_net.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        quick_print("Epoch {}/{}".format(epoch, num_epochs - 1))
        quick_print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                exp_lr_scheduler.step()
                poster_net.train(True)
            else:
                poster_net.train(False)

            running_loss = 0.0
            running_corrects = 0.0

            for data in loaders[phase]:
                img, label, title, imdb_id = data

                if use_gpu:
                    img = Variable(img.cuda())
                    label = Variable(label.cuda())
                else:
                    img, label = Variable(img), Variable(label)

                net_optimizer.zero_grad()

                scores = poster_net(img)
                _, preds = torch.max(scores, 1)
                loss = criterion(scores, label)

                if phase == "train":
                    loss.backward()
                    net_optimizer.step()

                running_loss += loss.data[0] * img.size(0)
                running_corrects += torch.sum(preds.data == label.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            quick_print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            if phase == "val":
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc)

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(poster_net.state_dict())

        quick_print("\n")

        torch.save(poster_net.state_dict(), folder_name + "/net_params")

        pickle_stat(val_losses, folder_name + "/val_losses.pkl")
        pickle_stat(val_accs, folder_name + "/val_accs.pkl")

    time_elapsed = time.time() - since
    quick_print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    quick_print("Best val Acc: {:4f}".format(best_acc))

    poster_net.load_state_dict(best_model_wts)

    torch.save(poster_net.state_dict(), folder_name + "/net_params")

if __name__ == '__main__':
    main()