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
        # parser.add_argument('--device', type=str, default='cuda:0', help='device')
        # parser.add_argument('--dim', type=int, default=32, help='embedding size')
        # parser.add_argument('--is_pre', type=bool, default=True, help='pretrain')

        # parser.add_argument('--dataset', type=str, default='book', help='dataset')
        # parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
        # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
        # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        # parser.add_argument('--epochs', type=int, default=50, help='epochs')
        # parser.add_argument('--device', type=str, default='cuda:0', help='device')
        # parser.add_argument('--dim', type=int, default=32, help='embedding size')
        # parser.add_argument('--is_pre', type=bool, default=True, help='pretrain')

        # parser.add_argument('--dataset', type=str, default='ml', help='dataset')
        # parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
        # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        # parser.add_argument('--epochs', type=int, default=100, help='epochs')
        # parser.add_argument('--device', type=str, default='cuda:0', help='device')
        # parser.add_argument('--dim', type=int, default=32, help='embedding size')
        # parser.add_argument('--is_pre', type=bool, default=True, help='pretrain')

        # parser.add_argument('--dataset', type=str, default='yelp', help='dataset')
        # parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
        # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
        # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        # parser.add_argument('--epochs', type=int, default=10, help='epochs')
        # parser.add_argument('--device', type=str, default='cuda:0', help='device')
        # parser.add_argument('--dim', type=int, default=32, help='embedding size')
        # parser.add_argument('--is_pre', type=bool, default=True, help='pretrain')

        parser.add_argument('--dataset', type=str, default='movie', help='dataset')
        parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
        parser.add_argument('--l2', type=float, default=1e-4, help='L2')
        parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        parser.add_argument('--epochs', type=int, default=50, help='epochs')
        parser.add_argument('--device', type=str, default='cuda:0', help='device')
        parser.add_argument('--dim', type=int, default=32, help='embedding size')
        parser.add_argument('--is_pre', type=bool, default=True, help='pretrain')

        args = parser.parse_args()
        indicators = train(args)
        recall_10_np[step] = indicators[0]
        ndcg_10_np[step] = indicators[1]

    print('Recall_10[\'Wide&Deep\'][\'' + args.dataset + '\'] =', recall_10_np.mean().round(4))
    print('NDCG_10[\'Wide&Deep\'][\'' + args.dataset + '\'] =', ndcg_10_np.mean().round(4))

'''
Recall_10['Wide&Deep']['music'] = 0.231
NDCG_10['Wide&Deep']['music'] = 0.2105

Recall_10['Wide&Deep']['book'] = 0.377
NDCG_10['Wide&Deep']['book'] = 0.2811

Recall_10['Wide&Deep']['ml'] = 0.0496
NDCG_10['Wide&Deep']['ml'] = 0.0719

Recall_10['Wide&Deep']['yelp'] = 0.415
NDCG_10['Wide&Deep']['yelp'] = 0.2875

Recall_10['Wide&Deep']['movie'] = 0.0919
NDCG_10['Wide&Deep']['movie'] = 0.0994

ml	[0.0399, 0.0495, 0.0295, 0.0399, 0.0301, 0.0297, 0.0297, 0.0376, 0.0496] , [0.0606, 0.0698, 0.0402, 0.0569, 0.0461, 0.0422, 0.0422, 0.0528, 0.0733]
'''