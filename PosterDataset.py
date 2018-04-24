import os
import random
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage

class PosterDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):

        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        imdb_id = self.data.iloc[idx, 0]
        img_name = os.path.join(self.root_dir, str(imdb_id) + ".jpg")
        img = Image.open(img_name, "r")
        img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.data.iloc[idx, 1].strip().split(" ")
        label = [int(num) for num in label]
        label = torch.LongTensor(label)
        title = self.data.iloc[idx, 2]
        return img, label, title, imdb_id

    def random_visualize(self, num):

        indices = []

        for i in range(num):
            indices.append(random.randint(0, len(self)))

        plt.rcParams["axes.titlesize"] = 4

        for i in range(num):
            idx = indices[i]

            img, label, title, imdb_id = self[idx]
            if not isinstance(img, Image.Image):
                img = ToPILImage()(img)

            print(img.size)

            ax = plt.subplot(1, num, i + 1)
            plt.tight_layout()
            plt.imshow(img)
            ax.set_title('ID #{}: \n{}'.format(imdb_id, title))
            ax.axis('off')

        plt.show()

# Test PosterDataset by randomly visualizing four of the posters.
def main():

    poster_dataset = PosterDataset(csv_file='data/labels.csv', root_dir='data/posters')
    poster_dataset.random_visualize(4)

if __name__ == '__main__':
    main()