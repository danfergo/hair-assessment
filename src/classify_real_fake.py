from torchvision import transforms as t
from torch.utils.data import DataLoader

import torch.optim as opt
import torch.nn as nn

from lib.event_listeners.exp_board.e_board import EBoard
from lib.metrics import accuracy
from lib.event_listeners.validator import Validator
from experimenter import experiment, run, e

from lib.event_listeners.logger import Logger
from lib.event_listeners.plotter import Plotter
from lib.train import train
from src.datasets.video_dataset import VideoDataset
from src.models.resnet50 import load_model as resnet50

import torch

torch.cuda.empty_cache()


def loader(partition, epoch_size):
    return DataLoader(
        VideoDataset(e.ws('data_path', partition),
                     epoch_size=epoch_size,
                     step_frames=e['step_frames'],
                     frame_transform=e['frame_transform'],
                     clip_len=e['clip_size']),
        batch_size=e['batch_size']
    )


@experiment(
    """
    I'm just testing the base resnet 3D and the whole script. 
    """,
    {
        'model': resnet50(),

        # data
        'data_path': './data/real_fake/',
        'clip_size': 8,
        'step_frames': 2,
        'frame_transform': t.Compose([
            t.Resize((224, 224)),
        ]),
        '{data_loader}': lambda: loader('train', e['batches_per_epoch']),
        '{val_loader}': lambda: loader('val', e['n_val_batches']),

        # train
        'loss': nn.CrossEntropyLoss(),
        '{optimizer}': lambda: opt.SGD(e['model'].parameters(), lr=0.01, momentum=0.9),
        'epochs': 50,
        'batches_per_epoch': 512,
        'batches_per_grad_update': 64,
        'batch_size': 1,
        'train_device': 'cuda',

        # validation
        'metrics': [accuracy],
        'metrics_names': ['Accuracy'],
        'n_val_batches': 5,
    },
    event_listeners=lambda: [
        Validator(),
        Logger(),
        Plotter(),
        EBoard()
    ]
)
def main():
    train()


run(main, open_e=False)
