import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def draw_param(params, indicators, xlabel, file, color, marker, metric):

    fig = plt.figure(figsize=[8, 6])
    ax = fig.add_subplot(111)
    plt.xlabel(xlabel, fontsize=25)
    plt.ylabel(metric, fontsize=25)
    plt.tick_params(labelsize=25)
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams["axes.unicode_minus"] = False
    plt.xticks(range(len(params)), params)
    # step = (max(indicators) - min(indicators)) / len(indicators)
    # plt.ylim(min(indicators), max(indicators))

    # plt.yticks(range(len(indicators)), indicators)
    ax.plot(range(len(params)),
             indicators,
             color=color,
             marker=marker,
             markerfacecolor='none')
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
    plt.grid()
    plt.savefig(file, bbox_inches='tight')


def draw_table(datasets, auc_dict):
    dataset_dict = {'music': 'Last.FM',
                    'book': 'Book-Crossing',
                    'ml': 'Movielens-100K',
                    'yelp': 'Yelp',
                    'movie': 'Movielens-1M'}
    for dataset in datasets:
        line = dataset_dict[dataset] + ' & '

        auc_np = auc_dict[dataset]

        for i in range(auc_np.shape[0]):
            if i == auc_np.shape[0] - 1:
                line += str(auc_np[i]) + ' \\\\ '
            else:
                line += str(auc_np[i]) + ' & '

        print(line)


if __name__ == '__main__':

    dims = [2, 4, 8, 16, 32, 64]
    Hs = [1, 2, 3, 4]
    Ks = [2, 4, 8, 16, 32, 64]

    dim_Recall_10_dict = {}

    dim_Recall_10_dict['music'] = np.array([0.2875, 0.2876, 0.2892, 0.2892, 0.2892, 0.2888])
    dim_Recall_10_dict['book'] = np.array([0.3782, 0.3792, 0.3802, 0.3795, 0.3796, 0.3784])
    dim_Recall_10_dict['ml'] = np.array([0.0574, 0.0588, 0.06, 0.0606, 0.0601, 0.0599])

    L_Recall_10_dict = {}

    L_Recall_10_dict['music'] = np.array([0.2891, 0.2887, 0.2891, 0.2884])
    L_Recall_10_dict['book'] = np.array([0.3805, 0.3805, 0.3788, 0.3775])
    L_Recall_10_dict['ml'] = np.array([0.0618, 0.0603, 0.0602, 0.06])


    K_Recall_10_dict = {}

    K_Recall_10_dict['music'] = np.array([0.2891, 0.2896, 0.2893, 0.2893, 0.2887, 0.2893])
    K_Recall_10_dict['book'] = np.array([0.38, 0.3799, 0.3796, 0.3803, 0.3808, 0.3799])
    K_Recall_10_dict['ml'] = np.array([0.059, 0.0595, 0.0608, 0.0606, 0.0609, 0.062])

    colors = ['r', 'g', 'b', 'purple', 'y', 'cyan',
              'orange', 'darkblue', 'darkgoldenrod', 'darkviolet', 'fuchsia', 'lightseagreen']
    markers = ['o', 's', 'v', 'x', 'D', '^', '<', '>', '+', '|', '*', 'd', '*']
    i = 0

    datasets = ['music', 'book', 'ml']
    print('L')
    draw_table(datasets, L_Recall_10_dict)
    print('K_l')
    draw_table(datasets, K_Recall_10_dict)
    print('dim')
    draw_table(datasets, dim_Recall_10_dict)
    # for dataset in ['ml']:
    #     draw_param(dims, dim_Recall_10_dict[dataset], '$d$', 'd-'+dataset+'.pdf', colors[i], markers[i], 'Recall@10')
    #     draw_param(Hs, L_Recall_10_dict[dataset], '$L$', 'L-'+dataset+'.pdf', colors[i], markers[i], 'Recall@10')
    #     draw_param(Ks, K_Recall_10_dict[dataset], '$K_l$', 'K_l-'+dataset+'.pdf', colors[i], markers[i], 'Recall@10')
    #     i += 1