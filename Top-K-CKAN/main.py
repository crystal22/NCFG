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
        # parser.add_argument('--lr', type=float, default=1e-1, help='learning rate')
        # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
        # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        # parser.add_argument('--epochs', type=int, default=50, help='epochs')
        # parser.add_argument('--device', type=str, default='cuda:0', help='device')
        # parser.add_argument('--dim', type=int, default=16, help='embedding size')
        # parser.add_argument('--L', type=int, default=2, help='H')
        # parser.add_argument('--K', type=int, default=8, help='K')
        # parser.add_argument('--agg', type=str, default='concat', help='K')

        # parser.add_argument('--dataset', type=str, default='book', help='dataset')
        # parser.add_argument('--lr', type=float, default=1e-1, help='learning rate')
        # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
        # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        # parser.add_argument('--epochs', type=int, default=50, help='epochs')
        # parser.add_argument('--device', type=str, default='cuda:0', help='device')
        # parser.add_argument('--dim', type=int, default=16, help='embedding size')
        # parser.add_argument('--L', type=int, default=2, help='H')
        # parser.add_argument('--K', type=int, default=8, help='K')
        # parser.add_argument('--agg', type=str, default='concat', help='K')

        # parser.add_argument('--dataset', type=str, default='ml', help='dataset')
        # parser.add_argument('--lr', type=float, default=1e-1, help='learning rate')
        # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
        # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        # parser.add_argument('--epochs', type=int, default=50, help='epochs')
        # parser.add_argument('--device', type=str, default='cuda:0', help='device')
        # parser.add_argument('--dim', type=int, default=16, help='embedding size')
        # parser.add_argument('--L', type=int, default=2, help='H')
        # parser.add_argument('--K', type=int, default=8, help='K')
        # parser.add_argument('--agg', type=str, default='concat', help='K')
        #
        # parser.add_argument('--dataset', type=str, default='yelp', help='dataset')
        # parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
        # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        # parser.add_argument('--epochs', type=int, default=20, help='epochs')
        # parser.add_argument('--device', type=str, default='cuda:0', help='device')
        # parser.add_argument('--dim', type=int, default=16, help='embedding size')
        # parser.add_argument('--L', type=int, default=2, help='H')
        # parser.add_argument('--K', type=int, default=8, help='K')
        # parser.add_argument('--agg', type=str, default='concat', help='K')

        parser.add_argument('--dataset', type=str, default='movie', help='dataset')
        parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
        parser.add_argument('--l2', type=float, default=1e-4, help='L2')
        parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
        parser.add_argument('--epochs', type=int, default=20, help='epochs')
        parser.add_argument('--device', type=str, default='cuda:0', help='device')
        parser.add_argument('--dim', type=int, default=16, help='embedding size')
        parser.add_argument('--L', type=int, default=2, help='H')
        parser.add_argument('--K', type=int, default=8, help='K')
        parser.add_argument('--agg', type=str, default='concat', help='K')

        parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')

        args = parser.parse_args()

        indicators = train(args)
        recall_10_np[step] = indicators[0]
        ndcg_10_np[step] = indicators[1]

    print('Recall_10[\'CKAN\'][\'' + args.dataset + '\'] =', recall_10_np.mean().round(4))
    print('NDCG_10[\'CKAN\'][\'' + args.dataset + '\'] =', ndcg_10_np.mean().round(4))

'''
Recall_10['CKAN']['music'] = 0.2393
NDCG_10['CKAN']['music'] = 0.2188

Recall_10['CKAN']['book'] = 0.2287
NDCG_10['CKAN']['book'] = 0.1531

Recall_10['CKAN']['ml'] = 0.0435
NDCG_10['CKAN']['ml'] = 0.0566

Recall_10['CKAN']['yelp'] = 0.2118
NDCG_10['CKAN']['yelp'] = 0.1217

Recall_10['CKAN']['movie'] = 0.1073
NDCG_10['CKAN']['movie'] = 0.1182

ml	[0.0378, 0.0438, 0.0442, 0.0436, 0.0442, 0.0425, 0.0429, 0.0459, 0.0451] , [0.0505, 0.0575, 0.0582, 0.0595, 0.0596, 0.0583, 0.0584, 0.0632, 0.0616]
'''