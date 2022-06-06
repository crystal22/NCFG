import time
from model import RippleNet
import numpy as np
import torch.nn as nn
import torch as t
import networkx as nx
import pandas as pd
import random
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import math
from data_loader import get_records, load_data
import copy


def train(args):
    data = load_data(args)
    n_entity, n_relation = data[0], data[1]
    train_set, test_records = data[2], data[3]
    rec, ripple_sets = data[4], data[5]

    model = RippleNet(args.dim, n_entity, args.H, n_relation, args.l1, args.l2)
    if t.cuda.is_available():
        model = model.to(args.device)

    optimizer = t.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    print(args.dataset + '-----------------------------------------')
    recall_list = []
    ndcg_list = []
    print('dim: %d' % args.dim, end='\t')
    print('H: %d' % args.H, end='\t')
    print('K: %d' % args.K, end='\t')
    print('lr: %1.0e' % args.lr, end='\t')
    print('l2: %1.0e' % args.l2, end='\t')
    print('batch_size: %d' % args.batch_size)
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

            users = [pair[0] for pair in pairs]
            predicts = model(pairs, ripple_sets)

            loss = model.computer_loss(labels, predicts, users, ripple_sets)

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
    print('Recall: %.4f \t NDCG: %.4f' % (max(recall_list), max(ndcg_list)))
    return max(recall_list), max(ndcg_list)





