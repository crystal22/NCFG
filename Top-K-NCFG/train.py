import torch as t
import torch.nn as nn
import torch.optim as optim
import time

from tqdm import tqdm

from model import NCFG
import math
from data_loader import get_records, load_data
import copy


def train(args):
    data = load_data(args)
    n_entity, n_user, n_relation = data[0], data[1], data[2]
    train_set, test_records = data[3], data[4]
    rec, ripple_sets = data[5], data[6]

    model = NCFG(args.dim, n_entity, n_user, n_relation, args.L, args.using_ncf)
    criterion = nn.BCELoss(reduction='sum')

    if t.cuda.is_available():
        model = model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    print(args.dataset + '-----------------------------------------')
    recall_list = []
    ndcg_list = []
    print('dim: %d' % args.dim, end='\t')
    print('L: %d' % args.L, end='\t')
    print('K_l: %d' % args.K_l, end='\t')
    print('lr: %1.0e' % args.lr, end='\t')
    print('l2: %1.0e' % args.l2, end='\t')
    print('batch_size: %d' % args.batch_size)
    for epoch in (range(args.epochs)):
        start = time.clock()
        cf_loss_sum = 0
        for i in range(0, len(train_set), args.batch_size):
            if (i + args.batch_size + 1) > len(train_set):
                batch_uvls = train_set[i:]
            else:
                batch_uvls = train_set[i: i + args.batch_size]

            pairs = [[uvl[0], uvl[1]] for uvl in batch_uvls]
            labels = t.tensor([int(uvl[2]) for uvl in batch_uvls]).view(-1).float()
            if t.cuda.is_available():
                labels = labels.to(args.device)
            predicts = model(pairs, ripple_sets)
            loss = criterion(predicts, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cf_loss_sum += loss.cpu().item()

        scores = model.get_scores(rec, ripple_sets)
        Recall, NDCG = model.topk_eval(scores, test_records, 10)
        ndcg_list.append(NDCG)
        recall_list.append(Recall)

        end = time.clock()
        print('epoch: %d \t Recall: %.4f NDCG: %.4f \t time: %d' % (
        (epoch + 1), Recall, NDCG, (end - start)))

    print(args.dataset + '\t Recall: %.4f \t NDCG: %.4f' % (max(recall_list), max(ndcg_list)))
    return max(recall_list), max(ndcg_list)