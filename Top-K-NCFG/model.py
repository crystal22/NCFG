import math

import numpy as np
import torch as t
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm


class NCF(nn.Module):

    def __init__(self, dim):
        super(NCF, self).__init__()
        self.l1 = nn.Linear(2*dim, dim)
        self.l2 = nn.Linear(dim, dim)
        self.l3 = nn.Linear(dim, dim)
        self.linear = nn.Linear(2*dim, 1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, user_embeddings, item_embeddings):
        user_item_embeddings = t.cat([user_embeddings, item_embeddings], dim=-1)
        y = self.l1(user_item_embeddings)
        y = self.dropout(y)
        y = t.relu(y)
        y = self.l2(y)
        y = self.dropout(y)
        y = t.relu(y)
        y = self.l3(y)
        y = self.dropout(y)
        y1 = t.sigmoid(y)

        y2 = user_embeddings * item_embeddings

        return t.sigmoid(self.linear(t.cat([y1, y2], dim=-1)))


class Aggregation(nn.Module):

    def __init__(self, input_dim, out_dim):
        super(Aggregation, self).__init__()
        self.W = nn.Linear(input_dim, out_dim)

    def forward(self, x):
        y = t.sigmoid(self.W(x))
        return y


class NCFG(nn.Module):

    def __init__(self, dim, n_entity, n_user, n_relation, L, using_ncf):

        super(NCFG, self).__init__()
        self.dim = dim
        embedding = t.randn(n_entity, dim)
        user_embedding_matrix = t.randn(n_user, dim)
        transformer_matrix = t.randn(n_relation, dim)
        nn.init.xavier_uniform_(embedding)
        nn.init.xavier_uniform_(transformer_matrix)
        self.embedding = nn.Parameter(embedding)
        self.user_embedding_matrix = nn.Parameter(user_embedding_matrix)
        self.transformer_matrix = nn.Parameter(transformer_matrix)
        self.L = L
        self.rnn = nn.RNN(dim, dim, num_layers=1)
        self.ncf = NCF(dim)
        self.using_ncf = using_ncf
        self.weight_attention = nn.Linear(3 * dim, 1)
        self.rnn.flatten_parameters()

    def forward(self, pairs, ripple_sets):

        users = [pair[0] for pair in pairs]
        items = [pair[1] for pair in pairs]

        heads_list, relations_list, tails_list = self.get_head_relation_and_tail(users, ripple_sets)
        user_embeddings = self.get_vector(users, heads_list, relations_list, tails_list)
        item_embeddings = self.embedding[items]
        if self.using_ncf:
            predict = self.ncf(user_embeddings, item_embeddings)
            return t.squeeze(predict, dim=1)
        else:
            predict = (user_embeddings * item_embeddings).sum(dim=1)
            return t.sigmoid(predict)

    def get_head_relation_and_tail(self, users, ripple_sets):

        heads_list = []
        relations_list = []
        tails_list = []
        for l in range(self.L):
            l_head_list = []
            l_relation_list = []
            l_tail_list = []

            for user in users:

                l_head_list.extend(ripple_sets[user][l][0])
                l_relation_list.extend(ripple_sets[user][l][1])
                l_tail_list.extend(ripple_sets[user][l][2])

            heads_list.append(l_head_list)
            relations_list.append(l_relation_list)
            tails_list.append(l_tail_list)

        return heads_list, relations_list, tails_list

    def get_vector(self, users, heads_list, relations_list, tails_list):

        o_list = []

        for l in range(self.L):
            head_embeddings = self.embedding[heads_list[l]].reshape(len(users), -1, self.dim)
            relation_embeddings = self.transformer_matrix[relations_list[l]].reshape(len(users), -1, self.dim)
            tail_embeddings = self.embedding[tails_list[l]].reshape(len(users), -1, self.dim)

            hRt = t.cat([head_embeddings, relation_embeddings, tail_embeddings], dim=-1)  # (batch_size, -1, 3 * dim)

            pi = t.sigmoid(self.weight_attention(hRt))  # (batch_size, -1, 1)
            pi = t.softmax(pi, dim=1)

            shape = [1, -1, self.dim]
            ht = t.cat([head_embeddings.reshape(shape), tail_embeddings.reshape(shape)], dim=0)
            triple_embeddings = self.rnn(ht)[0][-1].reshape(len(users), -1, self.dim)
            o_embeddings = (pi * triple_embeddings).sum(dim=1)
            o_list.append(o_embeddings)

        return sum(o_list)

    def get_scores(self, rec, ripple_sets):
        scores = {}
        self.eval()
        for user in (rec):
            items = list(rec[user])
            pairs = [[user, item] for item in items]
            predict = []
            for i in range(0, len(pairs), 1024):
                predict.extend(self.forward(pairs[i: i+1024], ripple_sets).cpu().detach().view(-1).numpy().tolist())
            # predict = self.forward(pairs, ripple_sets).cpu().detach().view(-1).numpy().tolist()
            n = len(pairs)
            user_scores = {items[i]: predict[i] for i in range(n)}
            user_list = list(dict(sorted(user_scores.items(), key=lambda x: x[1], reverse=True)).keys())
            scores[user] = user_list
        self.train()
        return scores

    def topk_eval(self, scores, test_records, K):

        recall_sum = 0
        ndcg_sum = 0
        for user in scores:
            rank_items = scores[user][:K]
            hit_num = len(set(rank_items) & set(test_records[user]))
            recall = hit_num / len(test_records[user])
            n = len(rank_items)
            a = sum([1 / math.log(i+2, 2) for i in range(n) if rank_items[i] in test_records[user]])
            b = sum([1 / math.log(i+2, 2) for i in range(len(test_records[user]))])
            ndcg = a / b

            recall_sum += recall
            ndcg_sum += ndcg

        Recall = recall_sum / len(scores)
        NDCG = ndcg_sum / len(scores)

        return Recall, NDCG

    def ctr_eval(self, data, ripple_sets, batch_size):
        self.eval()
        pred_label = []
        for i in range(0, len(data), batch_size):
            if (i + batch_size + 1) > len(data):
                batch_uvls = data[i:]
            else:
                batch_uvls = data[i: i + batch_size]

            pairs = [[uvl[0], uvl[1]] for uvl in batch_uvls]

            predicts = self.forward(pairs, ripple_sets).view(-1).cpu().detach().numpy().tolist()
            pred_label.extend(predicts)

        self.train()
        true_label = [line[2] for line in data]
        auc = roc_auc_score(true_label, pred_label)
        np_array = np.array(pred_label)
        np_array[np_array >= 0.5] = 1
        np_array[np_array < 0.5] = 0
        acc = accuracy_score(true_label, np_array.tolist())

        return auc, acc