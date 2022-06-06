import numpy as np

from train import train
import argparse


if __name__ == '__main__':

    steps = 5
    recall_10_np = np.zeros([steps])
    ndcg_10_np = np.zeros([steps])

    for step in range(steps):
        parser = argparse.ArgumentParser()

        # parser.add_argument('--dataset', type=str, default='music', help='dataset')
        # parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
        # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
        # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        # parser.add_argument('--epochs', type=int, default=20, help='epochs')
        # parser.add_argument("--device", type=str, default='cuda:3', help='device')
        # parser.add_argument('--dim', type=int, default=16, help='embedding size')
        # parser.add_argument('--n_neighbors', type=int, default=8, help='the number of neighbors')
        # parser.add_argument('--n_iter', type=int, default=2, help='the number of layers of KGCN')

        # parser.add_argument('--dataset', type=str, default='book', help='dataset')
        # parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
        # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
        # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        # parser.add_argument('--epochs', type=int, default=10, help='epochs')
        # parser.add_argument("--device", type=str, default='cuda:3', help='device')
        # parser.add_argument('--dim', type=int, default=16, help='embedding size')
        # parser.add_argument('--n_neighbors', type=int, default=8, help='the number of neighbors')
        # parser.add_argument('--n_iter', type=int, default=2, help='the number of layers of KGCN')
        #
        # parser.add_argument('--dataset', type=str, default='ml', help='dataset')
        # parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
        # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
        # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        # parser.add_argument('--epochs', type=int, default=50, help='epochs')
        # parser.add_argument("--device", type=str, default='cuda:3', help='device')
        # parser.add_argument('--dim', type=int, default=16, help='embedding size')
        # parser.add_argument('--n_neighbors', type=int, default=8, help='the number of neighbors')
        # parser.add_argument('--n_iter', type=int, default=2, help='the number of layers of KGCN')

        # parser.add_argument('--dataset', type=str, default='yelp', help='dataset')
        # parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
        # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
        # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        # parser.add_argument('--epochs', type=int, default=20, help='epochs')
        # parser.add_argument("--device", type=str, default='cuda:2', help='device')
        # parser.add_argument('--dim', type=int, default=16, help='embedding size')
        # parser.add_argument('--n_neighbors', type=int, default=8, help='the number of neighbors')
        # parser.add_argument('--n_iter', type=int, default=2, help='the number of layers of KGCN')
        #
        parser.add_argument('--dataset', type=str, default='movie', help='dataset')
        parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
        parser.add_argument('--l2', type=float, default=1e-4, help='L2')
        parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        parser.add_argument('--epochs', type=int, default=20, help='epochs')
        parser.add_argument("--device", type=str, default='cuda:2', help='device')
        parser.add_argument('--dim', type=int, default=16, help='embedding size')
        parser.add_argument('--n_neighbors', type=int, default=8, help='the number of neighbors')
        parser.add_argument('--n_iter', type=int, default=2, help='the number of layers of KGCN')

        parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')

        args = parser.parse_args()
        indicators = train(args)
        recall_10_np[step] = indicators[0]
        ndcg_10_np[step] = indicators[1]

    print('Recall_10[\'KGCN\'][\'' + args.dataset + '\'] =', recall_10_np.mean().round(4))
    print('NDCG_10[\'KGCN\'][\'' + args.dataset + '\'] =', ndcg_10_np.mean().round(4))


'''
Recall_10['KGCN']['music'] = 0.2721
NDCG_10['KGCN']['music'] = 0.2328

Recall_10['KGCN']['book'] = 0.3132
NDCG_10['KGCN']['book'] = 0.2004

Recall_10['KGCN']['ml'] = 0.0615
NDCG_10['KGCN']['ml'] = 0.0979

Recall_10['KGCN']['yelp'] = 0.3927
NDCG_10['KGCN']['yelp'] = 0.2614

Recall_10['KGCN']['movie'] = 0.1085
NDCG_10['KGCN']['movie'] = 0.1138
ml	[0.0339, 0.0415, 0.0486, 0.0541, 0.0577, 0.059, 0.0581, 0.0602, 0.0599] , [0.0437, 0.0537, 0.0628, 0.074, 0.0793, 0.0864, 0.0863, 0.0906, 0.0927]
'''


