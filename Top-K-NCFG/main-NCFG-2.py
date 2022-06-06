import numpy as np

from train import train
import argparse


if __name__ == '__main__':
    dims = [2, 4, 8, 16, 32, 64]
    Ls = [1, 2, 3, 4]
    Ks = [2, 4, 8, 16, 32, 64]
    ratios = [i*0.1 for i in range(1, 10)]
    ps = Ks
    dataset = 'book'
    n = len(ps)
    steps = 5
    recall_10_np = np.zeros([steps, n])
    ndcg_10_np = np.zeros([steps, n])
    for step in range(steps):

        for i in range(n):
            np.random.seed(555)
            parser = argparse.ArgumentParser()

            parser.add_argument('--dataset', type=str, default=dataset, help='dataset')
            parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
            parser.add_argument('--l2', type=float, default=1e-4, help='L2')
            parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
            parser.add_argument('--epochs', type=int, default=10, help='epochs')
            parser.add_argument('--device', type=str, default='cuda:0', help='device')
            parser.add_argument('--dim', type=int, default=16, help='embedding size')
            parser.add_argument('--L', type=int, default=2, help='L')
            parser.add_argument('--K_l', type=int, default=Ks[i], help='K_l')
            parser.add_argument('--using_ncf', type=bool, default=False, help='if using NCF?')
            parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')

            args = parser.parse_args()
            indicator = train(args)
            recall_10_np[step][i] = indicator[0]
            ndcg_10_np[step][i] = indicator[1]

    print(dataset, end='\t')
    print(recall_10_np.mean(axis=0).round(4).tolist(), ',', ndcg_10_np.mean(axis=0).round(4).tolist())


'''



'''

