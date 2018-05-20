import os
from torch.utils import data
from unidecode import unidecode
import pandas as pd

class NamesDataset(data.Dataset):

    def __init__(self, root_dir, train=True, transform=None, label_transform=None):
        super(NamesDataset, self).__init__()
        self.transform = (lambda x: x) if transform is None else transform
        self.label_transform = (lambda x: x) if label_transform is None else label_transform

        if train:
            filename =  os.path.join(root_dir, 'train.csv')
        else:
            filename =  os.path.join(root_dir, 'test.csv')

        self.df = pd.read_csv(filename)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[[idx]]
        lastname = item.LastName.values[0]
        origin = item.Origin.values[0]
        return self.transform(lastname), self.label_transform(origin)
