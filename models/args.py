import os
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description="PyTorch deep learning models for document classification")

    parser.add_argument('--no-cuda', action='store_false', dest='cuda')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=3435)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--data-dir', default=os.path.join(os.pardir, 'hedwig-data', 'datasets'))
    parser.add_argument('--metrics-json', type=str, default='metrics.json')
    parser.add_argument('--evaluate-test', action='store_true')
    parser.add_argument('--evaluate-dev', action='store_true')
    parser.add_argument('--binary-label', default='answer')
    parser.add_argument('--num-folds', type=int, default=4)
    parser.add_argument('--fold-num', type=int, default=-1)
    return parser
