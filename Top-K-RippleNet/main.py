import numpy as np

from train import train
import argparse


if __name__ == '__main__':

    steps = 5
    recall_10_np = np.zeros(steps)
    ndcg_10_np = np.zeros(steps)
    for step in range(steps):
        np.random.seed(555)
        parser = argparse.ArgumentParser()

        # parser.add_argument('--dataset', type=str, default='music', help='dataset')
        # parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
        # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
        # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        # parser.add_argument('--epochs', type=int, default=10, help='epochs')
        # parser.add_argument("--device", type=str, default='cuda:3', help='device')
        # parser.add_argument('--dim', type=int, default=16, help='embedding size')
        # parser.add_argument('--H', type=int, default=2, help='H')
        # parser.add_argument('--K', type=int, default=8, help='K')
        # parser.add_argument('--l1', type=float, default=1e-2, help='L1')
        # parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')

        # parser.add_argument('--dataset', type=str, default='book', help='dataset')
        # parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
        # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
        # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        # parser.add_argument('--epochs', type=int, default=10, help='epochs')
        # parser.add_argument("--device", type=str, default='cuda:1', help='device')
        # parser.add_argument('--dim', type=int, default=16, help='embedding size')
        # parser.add_argument('--H', type=int, default=2, help='H')
        # parser.add_argument('--K', type=int, default=8, help='K')
        # parser.add_argument('--l1', type=float, default=1e-2, help='L1')
        # parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')
        #
        # parser.add_argument('--dataset', type=str, default='ml', help='dataset')
        # parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
        # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
        # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        # parser.add_argument('--epochs', type=int, default=20, help='epochs')
        # parser.add_argument("--device", type=str, default='cuda:2', help='device')
        # parser.add_argument('--dim', type=int, default=16, help='embedding size')
        # parser.add_argument('--H', type=int, default=2, help='H')
        # parser.add_argument('--K', type=int, default=8, help='K')
        # parser.add_argument('--l1', type=float, default=1e-2, help='L1')
        # parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')
        #
        # parser.add_argument('--dataset', type=str, default='yelp', help='dataset')
        # parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
        # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
        # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        # parser.add_argument('--epochs', type=int, default=20, help='epochs')
        # parser.add_argument("--device", type=str, default='cuda:3', help='device')
        # parser.add_argument('--dim', type=int, default=16, help='embedding size')
        # parser.add_argument('--H', type=int, default=2, help='H')
        # parser.add_argument('--K', type=int, default=8, help='K')
        # parser.add_argument('--l1', type=float, default=1e-2, help='L1')
        # parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')

        parser.add_argument('--dataset', type=str, default='movie', help='dataset')
        parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
        parser.add_argument('--l2', type=float, default=1e-4, help='L2')
        parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        parser.add_argument('--epochs', type=int, default=20, help='epochs')
        parser.add_argument("--device", type=str, default='cuda:3', help='device')
        parser.add_argument('--dim', type=int, default=16, help='embedding size')
        parser.add_argument('--H', type=int, default=2, help='H')
        parser.add_argument('--K', type=int, default=8, help='K')
        parser.add_argument('--l1', type=float, default=1e-2, help='L1')
        parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')

        args = parser.parse_args()

        indicators = train(args)
        recall_10_np[step] = indicators[0]
        ndcg_10_np[step] = indicators[1]

    print('Recall_10[\'RippleNet\'][\'' + args.dataset + '\'] =', recall_10_np.mean().round(4))
    print('NDCG_10[\'RippleNet\'][\'' + args.dataset + '\'] =', ndcg_10_np.mean().round(4))


'''
Recall_10['RippleNet']['music'] = 0.2949
NDCG_10['RippleNet']['music'] = 0.2817

Recall_10['RippleNet']['book'] = 0.3679
NDCG_10['RippleNet']['book'] = 0.2673

Recall_10['RippleNet']['ml'] = 0.0484
NDCG_10['RippleNet']['ml'] = 0.0655

Recall_10['RippleNet']['yelp'] = 0.4453
NDCG_10['RippleNet']['yelp'] = 0.3086

Recall_10['RippleNet']['movie'] = 0.1253
NDCG_10['RippleNet']['movie'] = 0.1409

ml	[0.0402, 0.0466, 0.0474, 0.0466, 0.0468, 0.0479, 0.0465, 0.049, 0.0481] , [0.0532, 0.0625, 0.0627, 0.0641, 0.0623, 0.065, 0.0639, 0.0673, 0.0655]
'''

