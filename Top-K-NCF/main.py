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
        # parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
        # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        # parser.add_argument('--epochs', type=int, default=50, help='epochs')
        # parser.add_argument('--device', type=str, default='cuda:1', help='device')
        # parser.add_argument('--dim', type=int, default=32, help='embedding size')
        # parser.add_argument('--is_pre', type=bool, default=True, help='pretrain')

        # parser.add_argument('--dataset', type=str, default='book', help='dataset')
        # parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
        # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
        # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        # parser.add_argument('--epochs', type=int, default=50, help='epochs')
        # parser.add_argument('--device', type=str, default='cuda:1', help='device')
        # parser.add_argument('--dim', type=int, default=32, help='embedding size')
        # parser.add_argument('--is_pre', type=bool, default=True, help='pretrain')

        # parser.add_argument('--dataset', type=str, default='ml', help='dataset')
        # parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
        # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        # parser.add_argument('--epochs', type=int, default=50, help='epochs')
        # parser.add_argument('--device', type=str, default='cuda:1', help='device')
        # parser.add_argument('--dim', type=int, default=32, help='embedding size')
        # parser.add_argument('--is_pre', type=bool, default=True, help='pretrain')

        # parser.add_argument('--dataset', type=str, default='yelp', help='dataset')
        # parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
        # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
        # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        # parser.add_argument('--epochs', type=int, default=10, help='epochs')
        # parser.add_argument('--device', type=str, default='cuda:0', help='device')
        # parser.add_argument('--dim', type=int, default=32, help='embedding size')
        # parser.add_argument('--is_pre', type=bool, default=True, help='pretrain')
        #
        parser.add_argument('--dataset', type=str, default='movie', help='dataset')
        parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
        parser.add_argument('--l2', type=float, default=1e-4, help='L2')
        parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        parser.add_argument('--epochs', type=int, default=50, help='epochs')
        parser.add_argument('--device', type=str, default='cuda:0', help='device')
        parser.add_argument('--dim', type=int, default=32, help='embedding size')
        parser.add_argument('--is_pre', type=bool, default=True, help='pretrain')

        parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')

        args = parser.parse_args()
        indicators = train(args)
        recall_10_np[step] = indicators[0]
        ndcg_10_np[step] = indicators[1]

    print('Recall_10[\'CTR-NCF\'][\'' + args.dataset + '\'] =', recall_10_np.mean().round(4))
    print('NDCG_10[\'CTR-NCF\'][\'' + args.dataset + '\'] =', ndcg_10_np.mean().round(4))


'''
Recall_10['CTR-NCF']['music'] = 0.2587
NDCG_10['CTR-NCF']['music'] = 0.2315

Recall_10['CTR-NCF']['book'] = 0.1906
NDCG_10['CTR-NCF']['book'] = 0.1206

Recall_10['CTR-NCF']['ml'] = 0.0384
NDCG_10['CTR-NCF']['ml'] = 0.0493

Recall_10['CTR-NCF']['yelp'] = 0.4367
NDCG_10['CTR-NCF']['yelp'] = 0.2971

Recall_10['Wide&Deep']['movie'] = 0.092
NDCG_10['Wide&Deep']['movie'] = 0.0995
ml	[0.0342, 0.028, 0.0282, 0.0161, 0.0168, 0.028, 0.0316, 0.0449, 0.0371] , [0.0435, 0.0348, 0.0351, 0.0165, 0.0176, 0.0371, 0.0427, 0.0635, 0.0517]


'''