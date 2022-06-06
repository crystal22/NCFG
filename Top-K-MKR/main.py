import numpy as np

from train import train
import argparse


if __name__ == '__main__':
    steps = 5
    recall_10_np = np.zeros([steps])
    ndcg_10_np = np.zeros([steps])

    for step in range(steps):
        np.random.seed(555)
        parser = argparse.ArgumentParser()

        # parser.add_argument('--dataset', type=str, default='music', help='dataset')
        # parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
        # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
        # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        # parser.add_argument('--epochs', type=int, default=10, help='epochs')
        # parser.add_argument('--device', type=str, default='cuda:1', help='device')
        # parser.add_argument('--dim', type=int, default=32, help='embedding size')
        # parser.add_argument('--L', type=int, default=1, help='L')
        # parser.add_argument('--T', type=int, default=5, help='T')
        # parser.add_argument('--l1', type=float, default=1e-6, help='kg loss weight')

        # parser.add_argument('--dataset', type=str, default='book', help='dataset')
        # parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
        # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
        # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        # parser.add_argument('--epochs', type=int, default=10, help='epochs')
        # parser.add_argument('--device', type=str, default='cuda:1', help='device')
        # parser.add_argument('--dim', type=int, default=32, help='embedding size')
        # parser.add_argument('--L', type=int, default=1, help='L')
        # parser.add_argument('--T', type=int, default=5, help='T')
        # parser.add_argument('--l1', type=float, default=1e-6, help='kg loss weight')

        # parser.add_argument('--dataset', type=str, default='ml', help='dataset')
        # parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
        # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
        # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        # parser.add_argument('--epochs', type=int, default=10, help='epochs')
        # parser.add_argument('--device', type=str, default='cuda:1', help='device')
        # parser.add_argument('--dim', type=int, default=32, help='embedding size')
        # parser.add_argument('--L', type=int, default=1, help='L')
        # parser.add_argument('--T', type=int, default=5, help='T')
        # parser.add_argument('--l1', type=float, default=1e-6, help='kg loss weight')

        # parser.add_argument('--dataset', type=str, default='yelp', help='dataset')
        # parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
        # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
        # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        # parser.add_argument('--epochs', type=int, default=10, help='epochs')
        # parser.add_argument('--device', type=str, default='cuda:1', help='device')
        # parser.add_argument('--dim', type=int, default=32, help='embedding size')
        # parser.add_argument('--L', type=int, default=1, help='L')
        # parser.add_argument('--T', type=int, default=5, help='T')
        # parser.add_argument('--l1', type=float, default=1e-6, help='kg loss weight')
        #
        parser.add_argument('--dataset', type=str, default='movie', help='dataset')
        parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
        parser.add_argument('--l2', type=float, default=1e-4, help='L2')
        parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        parser.add_argument('--epochs', type=int, default=20, help='epochs')
        parser.add_argument('--device', type=str, default='cuda:1', help='device')
        parser.add_argument('--dim', type=int, default=32, help='embedding size')
        parser.add_argument('--L', type=int, default=1, help='L')
        parser.add_argument('--T', type=int, default=5, help='T')
        parser.add_argument('--l1', type=float, default=1e-6, help='kg loss weight')

        parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')

        args = parser.parse_args()
        indicators = train(args)
        recall_10_np[step] = indicators[0]
        ndcg_10_np[step] = indicators[1]

    print('Recall_10[\'MKR\'][\'' + args.dataset + '\'] =', recall_10_np.mean().round(4))
    print('NDCG_10[\'MKR\'][\'' + args.dataset + '\'] =', ndcg_10_np.mean().round(4))

'''
Recall_10['MKR']['music'] = 0.2916
NDCG_10['MKR']['music'] = 0.2608

Recall_10['MKR']['book'] = 0.3841
NDCG_10['MKR']['book'] = 0.2795

Recall_10['MKR']['ml'] = 0.062
NDCG_10['MKR']['ml'] = 0.0957

Recall_10['MKR']['yelp'] = 0.4419
NDCG_10['MKR']['yelp'] = 0.3018

Recall_10['MKR']['movie'] = 0.1268
NDCG_10['MKR']['movie'] = 0.1426
'''