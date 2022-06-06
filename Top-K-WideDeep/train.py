import numpy as np
import torch as t
import torch.nn as nn
import torch.optim as optim
import time

from tqdm import tqdm

from model import WideDeep
from data_loader import load_data, get_records
import copy


def train(args):
    data = load_data(args)
    pre_entity_embedding = data[0]
    n_user = data[1]
    train_set, test_records = data[2], data[3]
    rec = data[4]
    embedding_matrix = pre_entity_embedding
    model = WideDeep(2*args.dim, args.dim, embedding_matrix, n_user, args.is_pre)
    criterion = nn.BCELoss()

    if t.cuda.is_available():
        model = model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    print(args.dataset + '-----------------------------------------')

    print('dim: %d' % args.dim, end='\t')
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
            labels = t.tensor([uvl[2] for uvl in batch_uvls]).view(-1).float()
            if t.cuda.is_available():
                labels = labels.to(args.device)
            predicts = model(pairs)

            loss = criterion(predicts, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scores = model.get_scores(rec)

        Recall, NDCG = model.topk_eval(scores, test_records, 10)
        ndcg_list.append(NDCG)
        recall_list.append(Recall)

        recall_list.append(Recall)
        ndcg_list.append(NDCG)

        end = time.clock()
        print('epoch: %d \t Recall: %.4f NDCG: %.4f \t time: %d' % (
            (epoch + 1), Recall, NDCG, (end - start)))

    print(args.dataset + '\t Recall: %.4f \t NDCG: %.4f' % (
        max(recall_list), max(ndcg_list)))
    return max(recall_list), max(ndcg_list)

