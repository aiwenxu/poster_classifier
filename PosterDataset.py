#TODO

import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class PosterDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):

        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir, )

