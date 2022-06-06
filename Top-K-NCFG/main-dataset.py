from data_loader import load_ratings, load_kg
import pandas as pd


def get_attributes(dataset):
    print(dataset + '-----------------------')
    data_dir = '../data/' + dataset + '/'
    rating_np = load_ratings(data_dir)
    kg_dict, n_entity, n_relation = load_kg(data_dir)
    triple_np = pd.read_csv(data_dir + 'kg.txt', header=None, delimiter='\t').values
    n_triple = triple_np.shape[0]
    user_set = set(rating_np[:, 0])
    item_set = set(rating_np[:, 1])
    n_user = len(user_set)
    n_item = len(item_set)
    n_rating = rating_np.shape[0] / 2
    data_density = (n_rating / (n_user * n_item)) * 100

    print('n_user: %d' % n_user)
    print('n_item: %d' % n_item)
    print('n_interaction: %d' % n_rating)
    print('n_entity: %d' % n_entity)
    print('n_triple: %d' % n_triple)
    print('n_relation: %d' % n_relation)
    print('data density: %.4f' % data_density)

    print('-------------------------------------')


if __name__ == '__main__':

    get_attributes('ml')
    get_attributes('music')
    get_attributes('book')
    get_attributes('yelp')
    get_attributes('movie')