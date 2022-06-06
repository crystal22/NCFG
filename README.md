# NCFG

## 1. Enviroment
You can run the code in the following environment.

python == 3.7

torch == 1.10.1

pandas == 1.3.5

numpy == 1.21.5

tqdm == 4.62.3

sklearn == 0.0

## 2. File
ratings.txt is a user record file sampled by 1:1. The data in first column is user, the data in second column is item, 
and the data in third is interaction(0 represents negative sample and 1 represents positive sample).

kg.txt is the file of kg. The data in first column is head entity, the data in second column is tail entity, and the data in third is relation.

relation-list.txt is the relation mapping file. The data in first column is id and the data in second column is relation.

user-list.txt is the user mapping file. The data in first column is id and the data in second column is user.

entity-list.txt is the entity mapping file. The data in first column is id and the data in second column is entity.

music is Last.FM and is from 'Multi-Task Feature Learning for Knowledge Graph Enhanced Recommendation'(https://github.com/hwwang55/MKR); 
book is Book-Crossing and is from 'Multi-Task Feature Learning for Knowledge Graph Enhanced Recommendation'(https://github.com/hwwang55/MKR);
movie is Movielens and is from 'Multi-Task Feature Learning for Knowledge Graph Enhanced Recommendation'(https://github.com/hwwang55/MKR);
ml is Movielens-100K and is from 'Recurrent Knowledge Graph Embedding for Effective Recommendation'(https://github.com/sunzhuntu/Recurrent-Knowledge-Graph-Embedding);
yelp is Yelp and is from 'Recurrent Knowledge Graph Embedding for Effective Recommendation'(https://github.com/sunzhuntu/Recurrent-Knowledge-Graph-Embedding).

## 3. Run the code
For Wide&Deep and NCF, you need to run main.py in TransR to get the embeddings of entity and run main.py. You can directly run main.py for others.

## 4. welcome to correct
