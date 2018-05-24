import os
from torch.utils import data
import pandas as pd

class CommentsDataset(data.Dataset):

    def __init__(self, root_dir, train=True, transform=None, target_transform=None):
        super(CommentsDataset, self).__init__()
        self.transform = (lambda x: x) if transform is None else transform
        self.target_transform = (lambda x: x) if target_transform is None else target_transform

        if train:
            filename =  os.path.join(root_dir, 'train.csv')
        else:
            filename =  os.path.join(root_dir, 'test.csv')

        self.df = pd.read_csv(filename)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[[idx]]
        text = item.comment_text.values[0]
        target = (bool(item.toxic.values[0]),
                  bool(item.severe_toxic.values[0]),
                  bool(item.obscene.values[0]),
                  bool(item.threat.values[0]),
                  bool(item.insult.values[0]),
                  bool(item.identity_hate.values[0]))
        return self.transform(text), self.target_transform(target)
