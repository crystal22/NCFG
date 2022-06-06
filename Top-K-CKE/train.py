import time

import numpy as np

from data_loader import load_kg, get_hrtts, get_uvvs, get_records, load_data
from model import CKE
import torch as t
import torch.optim as optim
import math
import copy


def train(args):
    data = load_data(args)
    n_entity, n_user, n_item, n_relation = data[0], data[1], data[2], data[3]
    train_set, test_records = data[4], data[5]
    rec, kg_dict = data[6], data[7]
    hrtts = get_hrtts(kg_dict)
    model = CKE(n_entity, n_user, n_item, n_relation, args.dim)

    if t.cuda.is_available():
        model = model.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.l2)

    uvvs = get_uvvs(train_set)
    train_data = [hrtts, uvvs]
    print(args.dataset + '----------------------------')
    print('dim: %d' % args.dim, end='\t')
    print('learning_rate: %1.0e' % args.learning_rate, end='\t')
    print('l2: %1.0e' % args.l2, end='\t')
    print('batch_size: %d' % args.batch_size)
    recall_list = []
    ndcg_list = []

    for epoch in range(args.epochs):
        kg_loss = 0
        start = time.clock()
        size = len(train_data[0])
        start_index = 0
        while start_index < size:
            if start_index + args.batch_size <= size:
                hrtts = train_data[0][start_index: start_index + args.batch_size]
            else:
                hrtts = train_data[0][start_index:]
            loss = -model(hrtts, 'kg')
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            kg_loss += loss.cpu().item()

            start_index += args.batch_size

        start_index = 0
        size = len(train_data[-1])
        while start_index < size:
            if start_index + args.batch_size <= size:
                uvvs = train_data[-1][start_index: start_index + args.batch_size]
            else:
                uvvs = train_data[-1][start_index:]
            loss = -model(uvvs, 'cf')
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            kg_loss += loss.cpu().item()

            start_index += args.batch_size

        scores = model.get_scores(rec)

        Recall, NDCG = model.topk_eval(scores, test_records, 10)
        ndcg_list.append(NDCG)
        recall_list.append(Recall)

        recall_list.append(Recall)
        ndcg_list.append(NDCG)

        end = time.clock()
        print('epoch: %d \t Recall: %.4f NDCG: %.4f \t time: %d' % ((epoch + 1), Recall, NDCG, (end-start)))

    print(args.dataset + '\t Recall: %.4f \t NDCG: %.4f' % (max(recall_list), max(ndcg_list)))
    return max(recall_list), max(ndcg_list)



