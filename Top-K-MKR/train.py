import numpy as np
import torch as t
from tqdm import tqdm

from data_loader import get_hrtts, get_records
import torch.optim as optim
from model import MKR
import time
import math
from data_loader import load_data
import copy


def train(args):
    data = load_data(args)
    n_entity, n_user, n_item, n_relation = data[0], data[1], data[2], data[3]
    train_set, test_records = data[4], data[5]
    rec, kg_dict = data[6], data[7]
    hrtts = get_hrtts(kg_dict, n_item)
    model = MKR(args.dim, args.L, args.T, args.l1, n_entity, n_user, n_item, n_relation)
    if t.cuda.is_available():
        model = model.to(args.device)
    print(args.dataset + '-----------------------------------')
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    print('dim: %d' % args.dim, end='\t')
    print('L: %d' % args.L, end='\t')
    print('l1: %1.0e' % args.l1, end='\t')
    print('lr: %1.0e' % args.lr, end='\t')
    print('l2: %1.0e' % args.l2, end='\t')
    print('batch_size: %d' % args.batch_size)
    recall_list = []
    ndcg_list = []

    for epoch in (range(args.epochs)):
        start = time.clock()
        model.train()
        size = len(train_set)
        for j in range(model.T):
            for i in range(0, size, args.batch_size):
                next_i = min([size, i+args.batch_size])
                data = train_set[i: next_i]
                loss = model.cal_rs_loss(data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        size = len(hrtts)
        for i in range(0, size, args.batch_size):
            next_i = min([size, i + args.batch_size])
            data = hrtts[i: next_i]
            loss = model.cal_kg_loss(data)

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
        print('epoch: %d \t Recall: %.4f NDCG: %.4f \t time: %d' % ((epoch + 1), Recall, NDCG, (end - start)))

    print(args.dataset + '\t Recall: %.4f \t NDCG: %.4f' % (max(recall_list), max(ndcg_list)))
    return max(recall_list), max(ndcg_list)



