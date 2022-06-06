import numpy as np
import pandas as pd


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


def data_split(ratings_np, ratio):
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

        if ratio < 1:
            size = int(len(train_indices) * ratio)
            if size < 1:
                size = 1
            train_indices = np.random.choice(train_indices, size, replace=False)

        train_set.extend([user, pos_record[i], 1] for i in train_indices)
        train_set.extend([user, neg_record[i], 0] for i in train_indices)

        test_set.extend([user, pos_record[i], 1] for i in test_indices)
        test_set.extend([user, neg_record[i], 0] for i in test_indices)

    np.random.shuffle(train_set)
    np.random.shuffle(test_set)
    return train_set, test_set


def get_rec(train_records, test_records, item_set):

    rec = dict()
    for user in test_records:
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


def get_ripple_set(train_dict, kg_dict, H, size):

    ripple_set_dict = {user: [] for user in train_dict}

    for u in (train_dict):

        next_e_list = train_dict[u]

        for h in range(H):
            h_head_list = []
            h_relation_list = []
            h_tail_list = []
            for head in next_e_list:
                if head not in kg_dict:
                    continue
                for rt in kg_dict[head]:
                    relation = rt[0]
                    tail = rt[1]
                    h_head_list.append(head)
                    h_relation_list.append(relation)
                    h_tail_list.append(tail)

            if len(h_head_list) == 0:
                h_head_list = ripple_set_dict[u][-1][0]
                h_relation_list = ripple_set_dict[u][-1][1]
                h_tail_list = ripple_set_dict[u][-1][0]
            else:
                replace = len(h_head_list) < size
                indices = np.random.choice(len(h_head_list), size, replace=replace)
                h_head_list = [h_head_list[i] for i in indices]
                h_relation_list = [h_relation_list[i] for i in indices]
                h_tail_list = [h_tail_list[i] for i in indices]

            ripple_set_dict[u].append((h_head_list, h_relation_list, h_tail_list))

            next_e_list = ripple_set_dict[u][-1][2]

    return ripple_set_dict


def load_data(args):
    data_dir = '../data/' + args.dataset + '/'
    ratings_np = load_ratings(data_dir)
    train_set, test_set = data_split(ratings_np, args.ratio)

    train_records = get_records(train_set)
    test_records = get_records(test_set)

    user_set = set(ratings_np[:, 0])
    item_set = set(ratings_np[:, 1])

    rec = get_rec(train_records, test_records, item_set)
    kg_dict, n_entity, n_relation = load_kg(data_dir)
    n_entity = n_entity + len(user_set)
    ripple_sets = get_ripple_set(train_records, kg_dict, args.H, args.K)
    data = [n_entity, n_relation, train_set, test_records, rec, ripple_sets]

    return data