import numpy as np
import pandas as pd
import torch as t


def load_kg(file):

    edges = pd.read_csv(file + 'kg.txt', delimiter='\t', header=None).values
    kg_dict = {}
    relation_set = set()
    entity_set = set()
    for edge in edges:
        head = edge[0]
        tail = edge[1]
        relation = edge[2]

        if head not in kg_dict:
            kg_dict[head] = []
        kg_dict[head].append([relation, tail])

        entity_set.add(head)
        entity_set.add(tail)
        relation_set.add(relation)

    return kg_dict, len(entity_set), len(relation_set)


def load_ratings(data_dir):

    data_np = pd.read_csv(data_dir + 'ratings.txt', delimiter='\t', header=None).values

    return data_np


def convert_dict(ratings_np):

    positive_records = dict()
    negative_records = dict()

    for pair in ratings_np:
        user = pair[0]
        item = pair[1]
        label = pair[2]

        if label == 1:
            if user not in positive_records:
                positive_records[user] = []
            positive_records[user].append(item)
        else:
            if user not in negative_records:
                negative_records[user] = []
            negative_records[user].append(item)

    return positive_records, negative_records


def data_split(ratings_np):
    # print('data split...')

    positive_records, negative_records = convert_dict(ratings_np)
    train_set = []
    test_set = []
    for user in positive_records:
        pos_record = positive_records[user]
        neg_record = negative_records[user]
        size = len(pos_record)

        test_indices = np.random.choice(size, int(size * 0.3), replace=False)
        train_indices = list(set(range(size)) - set(test_indices))

        train_set.extend([user, pos_record[i], 1] for i in train_indices)
        train_set.extend([user, neg_record[i], 0] for i in train_indices)

        test_set.extend([user, pos_record[i], 1] for i in test_indices)
        test_set.extend([user, neg_record[i], 0] for i in test_indices)

    np.random.shuffle(train_set)
    np.random.shuffle(test_set)
    return train_set, test_set


def get_rec(train_records, test_records, item_set):

    rec = dict()
    users = list(test_records.keys())

    for user in users:
        neg_items = list(item_set - set(train_records[user]) - set(test_records[user]))
        rec[user] = np.random.choice(neg_items, 50).tolist() + np.random.choice(test_records[user], 1).tolist()
    return rec


def get_records(data_set):

    records = dict()

    for pair in data_set:
        user = pair[0]
        item = pair[1]
        label = pair[2]

        if label == 1:
            if user not in records:
                records[user] = []

            records[user].append(item)

    return records


def load_data(args):
    data_dir = '../data/' + args.dataset + '/'
    ratings_np = load_ratings(data_dir)
    train_set, test_set = data_split(ratings_np)

    train_records = get_records(train_set)

    test_records = get_records(test_set)

    item_set = set(ratings_np[:, 1])

    rec = get_rec(train_records, test_records, item_set)
    param_dict = t.load('./data/' + args.dataset + '/' + str(args.dim) + '_emb.pkl')
    pre_entity_embedding = param_dict['ent_emb'].view(-1, args.dim)
    n_user = len(set(ratings_np[:, 0]))
    data = [pre_entity_embedding, n_user, train_set, test_records, rec]

    return data