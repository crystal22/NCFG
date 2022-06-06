import time

from tqdm import tqdm

from model import KGCN
import numpy as np
import torch.nn as nn
import torch as t
import math
from data_loader import load_data, get_records
import copy


def train(args):
    data = load_data(args)
    n_entity, n_user, n_relation = data[0], data[1], data[2]
    train_set, test_records = data[3], data[4]
    rec, adj_entity_np, adj_relation_np = data[5], data[6], data[7]

    criterion = nn.BCELoss()
    model = KGCN(n_entity, n_user, n_relation, args.dim, args.n_iter, args.n_neighbors)
    if t.cuda.is_available():
        model = model.to(args.device)

    optimizer = t.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    print(args.dataset + '-----------------------------------------')

    print('dim: %d' % args.dim, end='\t')
    print('n_iter: %d' % args.n_iter, end='\t')
    print('n_neighbors: %d' % args.n_neighbors, end='\t')
    print('lr: %1.0e' % args.lr, end='\t')
    print('l2: %1.0e' % args.l2, end='\t')
    print('batch_size: %d' % args.batch_size)
    recall_list = []
    ndcg_list = []

    for epoch in (range(args.epochs)):

        start = time.clock()

        for i in range(0, len(train_set), args.batch_size):

            if (i + args.batch_size + 1) > len(train_set):
                batch_uvls = train_set[i:]
            else:
                batch_uvls = train_set[i: i + args.batch_size]

            pairs = [[uvl[0], uvl[1]] for uvl in batch_uvls]
            labels = t.tensor([int(uvl[2]) for uvl in batch_uvls]).view(-1).float()
            if t.cuda.is_available():
                labels = labels.to(args.device)

            predicts = model(pairs, adj_entity_np, adj_relation_np)

            loss = criterion(predicts, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scores = model.get_scores(rec, adj_entity_np, adj_relation_np)

        Recall, NDCG = model.topk_eval(scores, test_records, 10)
        ndcg_list.append(NDCG)
        recall_list.append(Recall)

        recall_list.append(Recall)
        ndcg_list.append(NDCG)

        end = time.clock()
        print('epoch: %d \t Recall: %.4f NDCG: %.4f \t time: %d' % (
            (epoch + 1), Recall, NDCG, (end - start)))

    print('Recall: %.4f \t NDCG: %.4f' % (max(recall_list), max(ndcg_list)))
    return max(recall_list), max(ndcg_list)
