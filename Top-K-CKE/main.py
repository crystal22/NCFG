import numpy as np

from train import train
import argparse


if __name__ == '__main__':
    steps = 5
    recall_10_np = np.zeros(steps)
    ndcg_10_np = np.zeros(steps)

    for step in range(steps):
        parser = argparse.ArgumentParser()

        # parser.add_argument('--dataset', type=str, default='music', help='dataset')
        # parser.add_argument('--learning_rate', type=float, default=5e-3, help='learning rate')
        # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
        # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        # parser.add_argument('--epochs', type=int, default=20, help='epochs')
        # parser.add_argument('--device', type=str, default='cuda:2', help='device')
        # parser.add_argument('--dim', type=int, default=32, help='embedding size')

        # parser.add_argument('--dataset', type=str, default='book', help='dataset')
        # parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
        # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
        # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        # parser.add_argument('--epochs', type=int, default=20, help='epochs')
        # parser.add_argument('--device', type=str, default='cuda:2', help='device')
        # parser.add_argument('--dim', type=int, default=32, help='embedding size')

        # parser.add_argument('--dataset', type=str, default='ml', help='dataset')
        # parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
        # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
        # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        # parser.add_argument('--epochs', type=int, default=50, help='epochs')
        # parser.add_argument('--device', type=str, default='cuda:1', help='device')
        # parser.add_argument('--dim', type=int, default=32, help='embedding size')

        # parser.add_argument('--dataset', type=str, default='yelp', help='dataset')
        # parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
        # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
        # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        # parser.add_argument('--epochs', type=int, default=20, help='epochs')
        # parser.add_argument('--device', type=str, default='cuda:1', help='device')
        # parser.add_argument('--dim', type=int, default=32, help='embedding size')

        parser.add_argument('--dataset', type=str, default='movie', help='dataset')
        parser.add_argument('--learning_rate', type=float, default=2e-3, help='learning rate')
        parser.add_argument('--l2', type=float, default=1e-4, help='L2')
        parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        parser.add_argument('--epochs', type=int, default=20, help='epochs')
        parser.add_argument('--device', type=str, default='cuda:2', help='device')
        parser.add_argument('--dim', type=int, default=32, help='embedding size')

        parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')

        args = parser.parse_args()
        indicators = train(args)
        recall_10_np[step] = indicators[0]
        ndcg_10_np[step] = indicators[1]

    print('Recall_10[\'CKE\'][\''+args.dataset+'\'] =', recall_10_np.mean().round(4))
    print('NDCG_10[\'CKE\'][\''+args.dataset+'\'] =', ndcg_10_np.mean().round(4))


'''     
Recall_10['CKE']['music'] = 0.2683
NDCG_10['CKE']['music'] = 0.2586

Recall_10['CKE']['book'] = 0.3102
NDCG_10['CKE']['book'] = 0.2332

Recall_10['CKE']['ml'] = 0.0608
NDCG_10['CKE']['ml'] = 0.0978

Recall_10['CKE']['yelp'] = 0.4243
NDCG_10['CKE']['yelp'] = 0.2932

Recall_10['CKE']['movie'] = 0.1233
NDCG_10['CKE']['movie'] = 0.1408
ml	[0.0249, 0.0403, 0.049, 0.0524, 0.0559, 0.0576, 0.0574, 0.0595, 0.0586] , [0.031, 0.0536, 0.0675, 0.0743, 0.0805, 0.0867, 0.0876, 0.0906, 0.0913]
'''