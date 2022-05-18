from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix

import numpy as np
import pandas as pd


def get_examples(dataframe, labels_column="rating"):
    examples = dataframe[['userid', 'itemid']].values
    labels = dataframe[f'{labels_column}'].values
    return examples, labels


def train_test_split(examples, labels, test_size=0.1, verbose=0):
    if verbose:
        print("Train/Test split ")
        print(100-test_size*100, "% of training data")
        print(test_size*100, "% of testing data")    

    train_examples, test_examples, train_labels, test_labels = sklearn_train_test_split(
        examples, 
        labels, 
        test_size=0.1, 
        random_state=42, 
        shuffle=True
    )

    train_users = train_examples[:, 0]
    test_users = test_examples[:, 0]

    train_items = train_examples[:, 1]
    test_items = test_examples[:, 1]

    x_train = np.array(list(zip(train_users, train_items)))
    x_test = np.array(list(zip(test_users, test_items)))

    y_train = train_labels
    y_test = test_labels

    if verbose:
        print()
        print('number of training examples : ', x_train.shape)
        print('number of training labels : ', y_train.shape)
        print('number of test examples : ', x_test.shape)
        print('number of test labels : ', y_test.shape)

    return (x_train, x_test), (y_train, y_test)


def ids_encoder(ratings):
    users = sorted(ratings['userid'].unique())
    items = sorted(ratings['itemid'].unique())

    uencoder = LabelEncoder()
    iencoder = LabelEncoder()

    uencoder.fit(users)
    iencoder.fit(items)

    ratings.userid = uencoder.transform(ratings.userid.tolist())
    ratings.itemid = iencoder.transform(ratings.itemid.tolist())

    return ratings, uencoder, iencoder


class ExplainableMatrixFactorization:
    
    
    def __init__(self, m, n, W, alpha=0.001, beta=0.01, lamb=0.1, k=10):
        self.W = W
        self.m = m
        self.n = n
        
        np.random.seed(64)
        
        self.k = k
        self.P = np.random.normal(size=(self.m,k))
        self.Q = np.random.normal(size=(self.n,k))
        
        self.alpha = alpha
        self.beta = beta
        self.lamb = lamb
        
        self.history = {
            "epochs":[],
            "loss":[],
            "val_loss":[],
        }
        
        
    def print_training_parameters(self):
        print('Training EMF')
        print(f'k={self.k} \t alpha={self.alpha} \t beta={self.beta} \t lambda={self.lamb}')
        
        
    def update_rule(self, u, i, error):
        self.P[u] = self.P[u] + \
            self.alpha*(2 * error*self.Q[i] - self.beta*self.P[u] - self.lamb*(self.P[u] - self.Q[i]) * self.W[u,i])
        
        self.Q[i] = self.Q[i] + \
            self.alpha*(2 * error*self.P[u] - self.beta*self.Q[i] + self.lamb*(self.P[u] - self.Q[i]) * self.W[u,i])
        
        
    def mae(self,  x_train, y_train):
        M = x_train.shape[0]
        error = 0
        for pair, r in zip(x_train, y_train):
            u, i = pair
            error += np.absolute(r - np.dot(self.P[u], self.Q[i]))
        return error/M
    
    
    def print_training_progress(self, epoch, epochs, error, val_error, steps=5):
        if epoch == 1 or epoch % steps == 0 :
                print(f"epoch {epoch}/{epochs} - loss : {round(error,3)} - val_loss : {round(val_error,3)}")
                
                
    def learning_rate_schedule(self, epoch, target_epochs = 20):
        if (epoch >= target_epochs) and (epoch % target_epochs == 0):
                factor = epoch // target_epochs
                self.alpha = self.alpha * (1 / (factor * 20))
                print("\nLearning Rate : {}\n".format(self.alpha))
        
        
    def fit(self, x_train, y_train, validation_data, epochs=10):
        self.print_training_parameters()
        
        x_test, y_test = validation_data
        
        for epoch in range(1, epochs+1):
            for pair, r in zip(x_train, y_train):                
                u,i = pair                
                r_hat = np.dot(self.P[u], self.Q[i])                
                e = r - r_hat
                self.update_rule(u, i, error=e)
                
            error = self.mae(x_train, y_train)
            val_error = self.mae(x_test, y_test)
            self.update_history(epoch, error, val_error)            
            self.print_training_progress(epoch, epochs, error, val_error, steps=1)
        
        return self.history
    
    
    def update_history(self, epoch, error, val_error):
        self.history['epochs'].append(epoch)
        self.history['loss'].append(error)
        self.history['val_loss'].append(val_error)
    
    
    def evaluate(self, x_test, y_test):
        error = self.mae(x_test, y_test)
        print(f"validation error : {round(error,3)}")
      
    
    def predict(self, userid, itemid, uencoder, iencoder):
        u = uencoder.transform([userid])[0]
        i = iencoder.transform([itemid])[0]
        r = np.dot(self.P[u], self.Q[i])
        return r
    
    
    def recommend(self, userid, uencoder, iencoder, N):
        u = uencoder.transform([userid])[0]

        predictions = np.dot(self.P[u], self.Q.T)

        top_idx = np.flip(np.argsort(predictions))[:N]

        top_items = iencoder.inverse_transform(top_idx)

        preds = predictions[top_idx]

        return top_items, preds
    
    
    def calc_mnap(self, k, df, users,uencoder, iencoder):
        if not 'itemid' in df.columns:
            errstr = "Column with items id's must be named itemid "
            raise TypeError(errstr)
        else:
            maps = []
            for user in users: 
                score = 0.0
                num_hits = 0.0
                top_items, preds = self.recommend(user, uencoder, iencoder,N = 25)
                rel_items = df[df.userid == user]['itemid'].tolist()
                for i,p in enumerate(top_items[:k]):
                    if p in rel_items and p not in top_items[:i]:
                        num_hits += 1.0
                        score += num_hits / (i+1.0)
                maps.append(score / min(len(rel_items), k))
            return np.mean(maps)

    
    def calc_recalls(self, k, df, users, uencoder, iencoder):
        if not 'itemid' in df.columns:
            errstr = "Column with items id's must be named itemid "
            raise TypeError(errstr)
        else:
            recals = []
            for user in users: 
                top_items, preds = self.recommend(user,uencoder, iencoder, N = 25)
                rel_items = df[df.userid == user]['itemid'].tolist()
                top_inter = 0
                for el in top_items[:5]:
                    if el in rel_items:
                        top_inter += 1
                recals.append(top_inter / len(top_items))
            return np.mean(recals)

    
    def dcg_at_k(self, r, k):
        r = r[:k]
        dcg = np.sum(r / np.log2(np.arange(2, len(r) + 2)))
        return dcg

    
    def ndgc_at_k(self, r, k, method = 0):
        dcg_max = self.dcg_at_k(sorted(r, reverse = True), k)
        return self.dcg_at_k(r, k) / dcg_max

    
    def calc_ndcg(self, predictions, k, users, ratings, movie_ids):
        ndgcs = []

        for target_users in np.unique(users):

            target_movie_id = movie_ids[target_users == users]
            target_rating = ratings[target_users == users]

            rel = target_rating[np.argsort(-predictions[target_users == users])]
            ndgc = self.ndgc_at_k(rel, k=k)
            ndgcs.append(ndgc)

        ndcg = np.mean(ndgcs)
        return ndcg