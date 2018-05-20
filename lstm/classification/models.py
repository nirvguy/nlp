import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

class LSTMClassifier(nn.Module):
    def __init__(self, embedd_dim, hidden_size, vocab_size, classes):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedd_dim)
        self.encoder = nn.LSTM(embedd_dim, hidden_size)
        self.linear = nn.Linear(hidden_size, classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, lens):
        x = self.embedding(x)
        x = pack_padded_sequence(x, lens, batch_first=True)
        outputs, hidden = self.encoder(x)
        outputs, _ = pad_packed_sequence(outputs)
        outputs = torch.stack([outputs[idx-1, i, :] for i, idx in zip(range(outputs.shape[1]), lens)])
        return self.linear(outputs)

    def predict(self, x, lens, k=1):
        x = self.softmax(self(x, lens))
        probas, indices = x.topk(k=k)
        probas = probas.data.cpu()
        indices = indices.data.cpu()
        return indices, probas
