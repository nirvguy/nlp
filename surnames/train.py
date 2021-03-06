import os
import argparse
import torch
from torch.utils.data import DataLoader
import torchtrainer
from torchtrainer.callbacks import ProgbarLogger, CSVLogger, ModelCheckpoint
from torchtrainer.meters import CategoricalAccuracy
from torchtrainer.utils.data import CrossFoldValidation
from .models import LSTMClassifier
from .dataset import NamesDataset
from .trainers import SupervisedTrainer
from .transforms import *
from . import config

BATCH_SIZE=100
LR=0.005
EPOCHS = 20
NUM_WORKERS = 0
TRAIN = True
TRAINING_PROPORTION = 0.8

DATA_PATH = os.path.join('data', 'surnames')
WEIGHTS_PATH = os.path.join('output', 'surnames', 'weights')
CSV_PATH = os.path.join('output', 'surnames', 'training_stats.csv')

def parse_args():
    parser = argparse.ArgumentParser(description='Train Surname classifier')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train', default=EPOCHS)
    parser.add_argument('--batch_size', type=int, help='Training batch size', default=BATCH_SIZE)
    parser.add_argument('--val_batch_size', type=int, help='Batch size', default=BATCH_SIZE)
    parser.add_argument('--logging_frecuency', type=float, help='Logging frecuency', default=0.3)
    parser.add_argument('--lr', type=float, help="Learning rate", default=LR)
    return parser.parse_args()

def main():
    args = parse_args()

    names_dataset = NamesDataset(DATA_PATH, train=True, transform=surname2tensor, label_transform=origin2tensor)

    splitter = CrossFoldValidation(names_dataset, valid_size=0.3)

    train_ds, val_ds = next(iter(splitter))

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, num_workers=NUM_WORKERS, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=args.val_batch_size, num_workers=NUM_WORKERS, collate_fn=collate_fn)

    model = LSTMClassifier(classes=len(idx2class), embedd_dim=config.EMBEDD, hidden_size=config.HIDDEN, vocab_size=len(alphabet))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    callbacks = [ProgbarLogger(),
                 ModelCheckpoint(path=WEIGHTS_PATH, monitor='val_acc', mode='max'),
                 CSVLogger(CSV_PATH)]

    trainer = SupervisedTrainer(model,
                      optimizer=optimizer,
                      criterion=criterion,
                      callbacks=callbacks,
                      acc_meters={'acc': CategoricalAccuracy()},
                      logging_frecuency=round(len(train_dl) * args.logging_frecuency))

    if config.USE_CUDA:
        trainer.cuda()

    trainer.train(train_dl, valid_dataloader=val_dl, epochs=args.epochs)

if __name__ == '__main__':
    main()
