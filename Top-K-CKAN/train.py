import torch as t
import torch.nn as nn
import torch.optim as optim
import random
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import time

from tqdm import tqdm

from data_loader import load_data, get_records
from model import CKAN
import math
import copy


def train(args):
    data = load_data(args)
    n_entity, n_relation = data[0], data[1]
    train_set, test_records = data[2], data[3]
    rec, ripple_sets = data[4], data[5]

    model = CKAN(n_entities=n_entity, dim=args.dim, n_relations=n_relation, L=args.L, agg=args.agg)

    if t.cuda.is_available():
        model = model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    criterion = nn.BCELoss()

    print(args.dataset + '-----------------------------------------')
    recall_list = []
    ndcg_list = []

    print('dim: %d' % args.dim, end='\t')
    print('L: %d' % args.L, end='\t')
    print('K: %d' % args.K, end='\t')
    print('lr: %1.0e' % args.lr, end='\t')
    print('l2: %1.0e' % args.l2, end='\t')
    print('batch_size: %d' % args.batch_size)
    for epoch in (range(args.epochs)):

        start = time.clock()
        model.train()
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
        scores = model.get_scores(rec, ripple_sets)

        Recall, NDCG = model.topk_eval(scores, test_records, 10)
        ndcg_list.append(NDCG)
        recall_list.append(Recall)

        recall_list.append(Recall)
        ndcg_list.append(NDCG)

        end = time.clock()
        print('epoch: %d \t Recall: %.4f NDCG: %.4f \t time: %d' % (
            (epoch + 1), Recall, NDCG, (end - start)))

    print(args.dataset + '\t Recall: %.4f \t NDCG: %.4f' % (max(recall_list), max(ndcg_list)))
    return max(recall_list), max(ndcg_list)





