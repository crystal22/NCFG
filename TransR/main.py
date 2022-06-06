import numpy as np

from train import train
import argparse


if __name__ == '__main__':

    for dataset in ['ml', 'music', 'movie', 'book', 'yelp']:
        for dim in [32]:
            np.random.seed(555)
            parser = argparse.ArgumentParser()
            parser.add_argument('--dataset', type=str, default=dataset, help='dataset')
            parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
            parser.add_argument('--l2', type=float, default=1e-4, help='L2')
            parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
            parser.add_argument('--epochs', type=int, default=100, help='epochs')
            parser.add_argument('--device', type=str, default='cuda:0', help='device')
            parser.add_argument('--dim', type=int, default=dim, help='embedding size')
            parser.add_argument('--C', type=int, default=0, help='C')
            args = parser.parse_args()

            train(args)
