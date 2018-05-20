import sys
import os
import time
import torch
import argparse
from torch import nn
from torch.autograd import Variable
from .models import LSTMClassifier
from .dataset import NamesDataset
from .transforms import surname2tensor, idx2class, alphabet
from . import config


HIDDEN = 60
EMBEDD = 15
WEIGHTS_PATH = os.path.join('output', 'surnames', 'weights', '0.pth')

class Predictor(object):
    def __init__(self, weights_path, k=1):
        self.weights_path = weights_path
        self.k = k
        self.load_model()

    def load_model(self):
        self.model = LSTMClassifier(classes=len(idx2class), embedd_dim=EMBEDD, hidden_size=HIDDEN, vocab_size=len(alphabet))
        self.model.load_state_dict(torch.load(self.weights_path))

    def __call__(self, surname):
        x = torch.LongTensor([surname2tensor(surname)])
        x = Variable(x)
        if config.USE_CUDA:
            x = x.cuda()
        indices, probas = self.model.predict(x, [len(surname)], k=self.k)
        indices, probas = indices[0].tolist(), (probas[0] * 100.0).tolist()
        labels = [idx2class[idx] for idx in indices]
        return zip(labels, probas)

def parse_args():
    parser = argparse.ArgumentParser(description='Train Surname classifier')
    parser.add_argument('--model-path', type=str, help='Model path', default=WEIGHTS_PATH)
    parser.add_argument('-k', type=int, help='Number of results', default=3)
    parser.add_argument('--only-labels', help='Output format', dest='only_labels', action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()

    predictor = Predictor(args.model_path, k=args.k)

    def format_only_labels(output):
        return surname + ", " + ", ".join("{}".format(l) for l, _ in output)

    def format_probas(output):
        return surname + ", " + ", ".join("{}: {:.3f}".format(l, p) for l, _ in output)

    if args.only_labels:
        _format = format_probas
    else:
        _format = format_only_labels

    for surname in sys.stdin:
        surname = surname.rstrip()
        output = predictor(surname)
        print(_format(output))

if __name__ == '__main__':
    main()
