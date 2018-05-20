import os
import argparse
import torch
from torch.utils.data import DataLoader
import torchtrainer
from torchtrainer.meters import CategoricalAccuracy, LossMeter
from torchtrainer.meters.aggregators.scale import percentage
from torchtrainer.meters.aggregators.batch import Average
from .models import LSTMClassifier
from .dataset import NamesDataset
from .trainers import SupervisedValidator
from .transforms import *
from . import config

BATCH_SIZE=100
NUM_WORKERS = 0

DATA_PATH = os.path.join('data', 'surnames')
WEIGHTS_PATH = os.path.join('output', 'surnames', 'weights', '0.pth')

def parse_args():
    parser = argparse.ArgumentParser(description='Train Surname classifier')
    parser.add_argument('--batch_size', type=int, help='Batch size', default=BATCH_SIZE)
    parser.add_argument('--model-path', type=str, help='Model path', default=WEIGHTS_PATH)
    parser.add_argument('--use-weights', type=bool, help='Use weights path to load model. Default: true', default=True)
    return parser.parse_args()

def main():
    args = parse_args()

    names_dataset = NamesDataset(DATA_PATH, train=False, transform=surname2tensor, label_transform=origin2tensor)

    dl = DataLoader(names_dataset, batch_size=args.batch_size, num_workers=NUM_WORKERS, collate_fn=collate_fn)

    model = LSTMClassifier(classes=len(idx2class), embedd_dim=config.EMBEDD, hidden_size=config.HIDDEN, vocab_size=len(alphabet))

    meters = {'Loss': LossMeter(torch.nn.CrossEntropyLoss()),
              'Accuracy': CategoricalAccuracy(aggregator=percentage(Average()))}

    validator = SupervisedValidator(model, meters=meters)

    if args.use_weights:
        model.load_state_dict(torch.load(args.model_path))

    if config.USE_CUDA:
        model.cuda()
        validator.cuda()

    metrics = validator.validate(dl)

    for name, value in metrics.items():
        print("{}: {:.3f}".format(name, value))

if __name__ == '__main__':
    main()
