import math

import numpy as np
import torch as t
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score


class DNN(nn.Module):

    def __init__(self, dim):
        # t.manual_seed(255)
        # t.cuda.manual_seed(255)
        super(DNN, self).__init__()
        self.l1 = nn.Linear(2*dim, 32)
        self.l2 = nn.Linear(32, 16)
        self.l3 = nn.Linear(16, 8)
        self.l4 = nn.Linear(8, 8)

    def forward(self, x):
        y = self.l1(x)
        y = t.relu(y)
        y = self.l2(y)
        y = t.relu(y)
        y = self.l3(y)
        y = t.relu(y)
        y = self.l4(y)
        y = t.relu(y)

        return y


class NCF(nn.Module):

    def __init__(self, dim, embedding_matrix, n_user, is_pre):
        # t.manual_seed(255)
        # t.cuda.manual_seed(255)
        super(NCF, self).__init__()
        self.dim = dim
        self.h = nn.Linear(dim + 8, 1)
        self.dnn = DNN(dim)

        gmf_embedding_matrix = t.randn(embedding_matrix.shape[0], dim)
        mlp_embedding_matrix = t.randn(embedding_matrix.shape[0], dim)
        gmf_user_embedding_matrix = t.randn(n_user, dim)
        mlp_user_embedding_matrix = t.randn(n_user, dim)

        nn.init.xavier_uniform_(gmf_user_embedding_matrix)
        nn.init.xavier_uniform_(mlp_user_embedding_matrix)
        nn.init.xavier_uniform_(gmf_embedding_matrix)
        nn.init.xavier_uniform_(mlp_embedding_matrix)

        if is_pre:
            self.gmf_embedding_matrix = nn.Parameter(embedding_matrix)
            self.mlp_embedding_matrix = nn.Parameter(embedding_matrix)
        else:
            self.gmf_embedding_matrix = nn.Parameter(gmf_embedding_matrix)
            self.mlp_embedding_matrix = nn.Parameter(mlp_embedding_matrix)

        self.gmf_user_embedding_matrix = nn.Parameter(gmf_user_embedding_matrix)
        self.mlp_user_embedding_matrix = nn.Parameter(mlp_user_embedding_matrix)

    def forward(self, pairs):
        users_id = [pair[0] for pair in pairs]
        items_id = [pair[1] for pair in pairs]
        gmf_user_embeddings = self.gmf_user_embedding_matrix[users_id]
        gmf_item_embeddings = self.gmf_embedding_matrix[items_id]
        mlp_user_embeddings = self.mlp_user_embedding_matrix[users_id]
        mlp_item_embeddings = self.mlp_embedding_matrix[items_id]
        y1 = self.gmf(gmf_user_embeddings, gmf_item_embeddings)
        y2 = self.mlp(mlp_user_embeddings, mlp_item_embeddings)

        y = t.cat([y1, y2], dim=1)

        predicts = t.sigmoid(self.h(y)).view(-1)

        return predicts

    def gmf(self, user_embeddings, item_embeddings):

        return user_embeddings * item_embeddings

    def mlp(self, user_embeddings, item_embeddings):

        x = t.cat([user_embeddings, item_embeddings], dim=1)

        return self.dnn(x)

    def get_scores(self, rec):
        scores = {}

        for u in (rec):
            pairs = [[u, item] for item in rec[u]]
            predict_np = self.forward(pairs).cpu().detach().numpy()
            n = predict_np.shape[0]
            user_scores = {rec[u][i]: predict_np[i] for i in range(n)}
            user_list = list(dict(sorted(user_scores.items(), key=lambda x: x[1], reverse=True)).keys())
            scores[u] = user_list

        return scores

    def ctr_eval(self, data, batch_size):
        self.eval()
        true_labels = [i[2] for i in data]
        pred_labels = []
        for i in range(0, len(data), batch_size):
            next_i = min([i + batch_size, len(data)])
            predict = self.forward(data[i: next_i]).cpu().detach().numpy()
            pred_labels.extend(predict.tolist())
        self.train()

        pred = np.array(pred_labels)
        auc = roc_auc_score(true_labels, pred)
        # auc = round(auc, 4)
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        acc = accuracy_score(true_labels, pred)
        # acc = round(acc, 4)

        return auc, acc

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

