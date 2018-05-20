import torch
from torch.utils.data.dataloader import default_collate

idx2class = ['Arabic',
           'Chinese',
           'Czech',
           'Dutch',
           'English',
           'French',
           'German',
           'Greek',
           'Irish',
           'Italian',
           'Japanese',
           'Korean',
           'Polish',
           'Portuguese',
           'Russian',
           'Scottish',
           'Spanish',
           'Vietnamese']

class2idx = { key: i for i, key in enumerate(idx2class) }
alphabet = ['<PAD>', '<UNK>'] + list('abcdefghijklmnopqrstuvwxyzßàáãäçèéêìíñòóõöùúüąłńśż')
char2idx = { key: i for i, key in enumerate(alphabet) }

def surname2tensor(surname):
    return list(map(lambda x: char2idx.get(x, 1), surname.lower()))

def origin2tensor(origin):
    return class2idx[origin]

def collate_fn(data):
    data.sort(key=lambda x: len(x[0]), reverse=True)
    max_length = len(data[0][0])
    #print(max_length)

    def pad_sequence(x):
        text, category = x
        return text + [0] * max(0, max_length - len(text)), len(text), category

    data = list(map(pad_sequence, data))
    texts, lengths, categories = zip(*data)
    assert(len(texts) == len(lengths) == len(categories))
    texts = torch.LongTensor(texts)
    categories = torch.LongTensor(categories)
    return texts, lengths, categories

def words2tensor(text):
    strs = re.split('(\W+)', text)
    return []
