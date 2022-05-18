try:
    __import__('implicit')
except ImportError:
    pip.main(['install', 'implicit'])   
    
    
try:
    __import__('recommenders')
except ImportError:
    pip.main(['install', 'recommenders'])
    
from implicit.evaluation import ndcg_at_k as ndcg_imp
from implicit.evaluation import mean_average_precision_at_k
from implicit.evaluation import precision_at_k as pr_imp
from recommenders.evaluation.python_evaluation import (ndcg_at_k, precision_at_k, 
                                                     recall_at_k, get_top_k_items)
from recommenders.models.surprise.surprise_utils import predict, compute_ranking_predictions


def eval_als(model,train, test):
    return print("-------- K = 1 --------",
                  "NDCG@k:\t%f" % ndcg_imp(model,train, test, K=1),
                  "Recall@k:\t%f" % pr_imp(model,train, test, K=1),
                  "MNAP@K:\t%f" % mean_average_precision_at_k(model,train, test, K=1), 
                  "-------- K = 5 --------",
                  "NDCG@k:\t%f" % ndcg_imp(model,train, test, K=5),
                  "Recall@k:\t%f" % pr_imp(model,train, test, K=5),
                  "MNAP@K:\t%f" % mean_average_precision_at_k(model,train, test, K=5), 
                  "-------- K = 10 --------",
                  "NDCG@k:\t%f" % ndcg_imp(model,train, test, K=10),
                  "Recall@k:\t%f" % pr_imp(model,train, test, K=10),
                  "MNAP@K:\t%f" % mean_average_precision_at_k(model,train, test, K=10), 
                  "-------- K = 15 --------",
                  "NDCG@k:\t%f" % ndcg_imp(model,train, test, K=15),
                  "Recall@k:\t%f" % pr_imp(model,train, test, K=15),
                  "MNAP@K:\t%f" % mean_average_precision_at_k(model,train, test, K=15), 
                  "-------- K = 20 --------",
                  "NDCG@k:\t%f" % ndcg_imp(model,train, test, K=20),
                  "Recall@k:\t%f" % pr_imp(model,train, test, K=20),
                  "MNAP@K:\t%f" % mean_average_precision_at_k(model,train, test, K=20), sep = '\n')
    
    
def eval_bpr(model,train, test):
    return print("-------- K = 1 --------",
                  "NDCG@k:\t%f" % ndcg_imp(model,train, test, K=1),
                  "Recall@k:\t%f" % pr_imp(model,train, test, K=1),
                  "MNAP@K:\t%f" % mean_average_precision_at_k(model,train, test, K=1), 
                  "-------- K = 5 --------",
                  "NDCG@k:\t%f" % ndcg_imp(model,train, test, K=5),
                  "Recall@k:\t%f" % pr_imp(model,train, test, K=5),
                  "MNAP@K:\t%f" % mean_average_precision_at_k(model,train, test, K=5), 
                  "-------- K = 10 --------",
                  "NDCG@k:\t%f" % ndcg_imp(model,train, test, K=10),
                  "Recall@k:\t%f" % pr_imp(model,train, test, K=10),
                  "MNAP@K:\t%f" % mean_average_precision_at_k(model,train, test, K=10), 
                  "-------- K = 15 --------",
                  "NDCG@k:\t%f" % ndcg_imp(model,train, test, K=15),
                  "Recall@k:\t%f" % pr_imp(model,train, test, K=15),
                  "MNAP@K:\t%f" % mean_average_precision_at_k(model,train, test, K=15), 
                  "-------- K = 20 --------",
                  "NDCG@k:\t%f" % ndcg_imp(model,train, test, K=20),
                  "Recall@k:\t%f" % pr_imp(model,train, test, K=20),
                  "MNAP@K:\t%f" % mean_average_precision_at_k(model,train, test, K=20), sep = '\n')
    
    

def eval_svd(model,train, test, usercol, itemcol):
    all_predictions = compute_ranking_predictions(model, train, usercol=usercol, itemcol=itemcol, remove_seen=True)
    return print("-------- K = 1 --------",
                  "NDCG@k:\t%f" % ndcg_at_k(test, all_predictions, col_user = usercol, col_item = itemcol, col_rating = 'Rating', col_prediction='prediction', k=1),
                  "Recall@k:\t%f" % precision_at_k(test, all_predictions, col_user = usercol, col_item = itemcol, col_rating = 'Rating', col_prediction='prediction', k=1),
                  "MNAP@K:\t%f" % mean_average_precision_at_k(model,train, test, K=1), 
                  "-------- K = 5 --------",
                  "NDCG@k:\t%f" % ndcg_at_k(test, all_predictions, col_user = usercol, col_item = itemcol, col_rating = 'Rating', col_prediction='prediction', k=5),
                  "Recall@k:\t%f" % precision_at_k(test, all_predictions, col_user = usercol, col_item = itemcol, col_rating = 'Rating', col_prediction='prediction', k=5),
                  "MNAP@K:\t%f" % mean_average_precision_at_k(model,train, test, K=5), 
                  "-------- K = 10 --------",
                  "NDCG@k:\t%f" % ndcg_at_k(test, all_predictions, col_user = usercol, col_item = itemcol, col_rating = 'Rating', col_prediction='prediction', k=10),
                  "Recall@k:\t%f" % precision_at_k(test, all_predictions, col_user = usercol, col_item = itemcol, col_rating = 'Rating', col_prediction='prediction', k=10),
                  "MNAP@K:\t%f" % mean_average_precision_at_k(model,train, test, K=10), 
                  "-------- K = 15 --------",
                  "NDCG@k:\t%f" % ndcg_at_k(test, all_predictions, col_user = usercol, col_item = itemcol, col_rating = 'Rating', col_prediction='prediction', k=15),
                  "Recall@k:\t%f" % precision_at_k(test, all_predictions, col_user = usercol, col_item = itemcol, col_rating = 'Rating', col_prediction='prediction', k=15),
                  "MNAP@K:\t%f" % mean_average_precision_at_k(model,train, test, K=15), 
                  "-------- K = 20 --------",
                  "NDCG@k:\t%f" % ndcg_at_k(test, all_predictions, col_user = usercol, col_item = itemcol, col_rating = 'Rating', col_prediction='prediction', k=20),
                  "Recall@k:\t%f" % precision_at_k(test, all_predictions, col_user = usercol, col_item = itemcol, col_rating = 'Rating', col_prediction='prediction', k=20),
                  "MNAP@K:\t%f" % mean_average_precision_at_k(model,train, test, K=20), sep = '\n')
    
def eval_lightfm(model,train, test):
    return 0 


def evaluate_prediction(predictions):
    ndgcs = []

    for target_users in np.unique(test_user_ids):
        
        target_movie_id = test_movie_ids[target_users == test_user_ids]
        target_rating = test_ratings[target_users == test_user_ids]

        rel = target_rating[np.argsort(-predictions[target_users == test_user_ids])]
        ndgc = ndgc_at_k(rel, k=30)
        ndgcs.append(ndgc)

    ndcg = np.mean(ndgcs)
    return ndcg

def dcg_at_k(r, k):
    r = r[:k]
    dcg = np.sum(r / np.log2(np.arange(2, len(r) + 2)))
    return dcg


def ndgc_at_k(r, k, method = 0):
    dcg_max = dcg_at_k(sorted(r, reverse = True), k)
    return dcg_at_k(r, k) / dcg_max


def eval_emf(model,x_test,y_test,predictions):
    test_user_ids = np.array([a for a,b in x_test]).astype(int)
    test_movie_ids = np.array([b for a,b in x_test]).astype(int)
    test_ratings = y_test
    ndgcs = []
    
    for target_users in np.unique(test_user_ids):
        
        target_movie_id = test_movie_ids[target_users == test_user_ids]
        target_rating = test_ratings[target_users == test_user_ids]

        rel = target_rating[np.argsort(-predictions[target_users == test_user_ids])]
        ndgc1 = ndgc_at_k(rel, k=1)
        ndgc5 = ndgc_at_k(rel, k=5)
        ndgc10 = ndgc_at_k(rel, k=10)
        ndgc15 = ndgc_at_k(rel, k=15)
        ndgc20 = ndgc_at_k(rel, k=20)

        ndgcs1.append(ndgc1)
        ndgcs5.append(ndgc5)
        ndgcs10.append(ndgc10)
        ndgcs15.append(ndgc15)
        ndgcs20.append(ndgc20)

    ndcg_f1 = np.mean(ndgcs1)
    ndcg_f5 = np.mean(ndgcs5)
    ndcg_f10 = np.mean(ndgcs10)
    ndcg_f15 = np.mean(ndgcs15)
    ndcg_f20 = np.mean(ndgcs20)

    return print("-------- K = 1 --------",
                  "NDCG@k:\t%f" % ndcg_f1,
                  "Recall@k:\t%f" % 0,
                  "MNAP@K:\t%f" % 0, 
                  "-------- K = 5 --------",
                  "NDCG@k:\t%f" % ndcg_f5,
                  "Recall@k:\t%f" % 0,
                  "MNAP@K:\t%f" % 0, 
                  "-------- K = 10 --------",
                  "NDCG@k:\t%f" % ndcg_f10,
                  "Recall@k:\t%f" % 0,
                  "MNAP@K:\t%f" % 0, 
                  "-------- K = 15 --------",
                  "NDCG@k:\t%f" % ndcg_f15,
                  "Recall@k:\t%f" % 0,
                  "MNAP@K:\t%f" % 0, 
                  "-------- K = 20 --------",
                  "NDCG@k:\t%f" % ndcg_f20,
                  "Recall@k:\t%f" % 0,
                  "MNAP@K:\t%f" % 0, sep = '\n')