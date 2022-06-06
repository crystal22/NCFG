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

        parser.add_argument('--dataset', type=str, default='music', help='dataset')
        parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
        parser.add_argument('--l2', type=float, default=1e-4, help='L2')
        parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        parser.add_argument('--epochs', type=int, default=10, help='epochs')
        parser.add_argument('--device', type=str, default='cuda:0', help='device')
        parser.add_argument('--dim', type=int, default=16, help='embedding size')
        parser.add_argument('--L', type=int, default=2, help='L')
        parser.add_argument('--K_l', type=int, default=8, help='K_l')
        parser.add_argument('--using_ncf', type=bool, default=False, help='if using NCF?')

        # parser.add_argument('--dataset', type=str, default='book', help='dataset')
        # parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
        # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
        # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        # parser.add_argument('--epochs', type=int, default=10, help='epochs')
        # parser.add_argument('--device', type=str, default='cuda:0', help='device')
        # parser.add_argument('--dim', type=int, default=16, help='embedding size')
        # parser.add_argument('--L', type=int, default=2, help='L')
        # parser.add_argument('--K_l', type=int, default=8, help='K_l')
        # parser.add_argument('--using_ncf', type=bool, default=False, help='if using NCF?')

        # parser.add_argument('--dataset', type=str, default='ml', help='dataset')
        # parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
        # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
        # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        # parser.add_argument('--epochs', type=int, default=30, help='epochs')
        # parser.add_argument('--device', type=str, default='cuda:0', help='device')
        # parser.add_argument('--dim', type=int, default=16, help='embedding size')
        # parser.add_argument('--L', type=int, default=2, help='L')
        # parser.add_argument('--K_l', type=int, default=8, help='K_l')
        # parser.add_argument('--using_ncf', type=bool, default=False, help='if using NCF?')

        # parser.add_argument('--dataset', type=str, default='yelp', help='dataset')
        # parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
        # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
        # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        # parser.add_argument('--epochs', type=int, default=20, help='epochs')
        # parser.add_argument('--device', type=str, default='cuda:0', help='device')
        # parser.add_argument('--dim', type=int, default=16, help='embedding size')
        # parser.add_argument('--L', type=int, default=2, help='L')
        # parser.add_argument('--K_l', type=int, default=8, help='K_l')
        # parser.add_argument('--using_ncf', type=bool, default=False, help='if using NCF?')

        # parser.add_argument('--dataset', type=str, default='movie', help='dataset')
        # parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
        # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
        # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        # parser.add_argument('--epochs', type=int, default=20, help='epochs')
        # parser.add_argument('--device', type=str, default='cuda:1', help='device')
        # parser.add_argument('--dim', type=int, default=16, help='embedding size')
        # parser.add_argument('--L', type=int, default=2, help='L')
        # parser.add_argument('--K_l', type=int, default=8, help='K_l')
        # parser.add_argument('--using_ncf', type=bool, default=False, help='if using NCF?')

        parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')

        args = parser.parse_args()

        indicators = train(args)
        recall_10_np[step] = indicators[0]
        ndcg_10_np[step] = indicators[1]

    print('Recall_10[\'NCFG\'][\''+args.dataset+'\'] =', recall_10_np.mean().round(4))
    print('NDCG_10[\'NCFG\'][\''+args.dataset+'\'] =', ndcg_10_np.mean().round(4))




'''
Recall_10['NCFG']['music'] = 0.2888
NDCG_10['NCFG']['music'] = 0.2613

Recall_10['NCFG']['book'] = 0.379
NDCG_10['NCFG']['book'] = 0.283

Recall_10['NCFG']['ml'] = 0.0607
NDCG_10['NCFG']['ml'] = 0.0905

L: ml	[0.0618, 0.0603, 0.0602, 0.06]
K: ml	[0.059, 0.0595, 0.0608, 0.0609, 0.062]
dim: ml	[0.0574, 0.0588, 0.06, 0.0606, 0.0601, 0.0599]

L: music [0.2891, 0.2887, 0.2891, 0.2884]
K: music [0.2891, 0.2896, 0.2893, 0.2893, 0.2887, 0.2893]
dim: music	[0.2875, 0.2876, 0.2892, 0.2892, 0.2892, 0.2888]

L: book	[0.3805, 0.3805, 0.3788, 0.3775]
K: book [0.38, 0.3799, 0.3796, 0.3803, 0.3808, 0.3799]
dim: book	[0.3782, 0.3792, 0.3802, 0.3795, 0.3796, 0.3784]
'''

