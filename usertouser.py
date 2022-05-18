from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from datetime import datetime
from glob import glob

from EMF import ids_encoder

import pandas as pd
import numpy as np
import os


class UserToUser:

    def __init__(self, ratings, movies, k=10, predictions_dir='predictions/user2user', metric='euclidean'):

        if metric not in ['cosine', 'euclidean']:
            raise Exception('UnknownSimilarityMetric : The similarity metric must be selected among '
                            'the followings : cosine, euclidean. You choosed {}'.format(metric))

        self.ratings, self.uencoder, self.iencoder = ids_encoder(ratings)
        self.means, self.ratings = self.prepare_ratings()
        self.ratings_matrix = self.create_ratings_matrix()
        self.k = k
        self.metric = metric
        self.model = self.init_similarity_model()
        self.predictions_dir = predictions_dir
        self.similarities, self.neighbors = self.compute_nearest_neighbors()
        self.movies = movies

        self.np_ratings = self.ratings.to_numpy()

        os.makedirs(self.predictions_dir, exist_ok=True)
        print('User to user recommendation model created with success!')

    def create_ratings_matrix(self):
        return csr_matrix(
            pd.crosstab(self.ratings.userid, self.ratings.itemid, self.ratings.rating, aggfunc=sum).fillna(0).values
        )

    def init_similarity_model(self):
        print('Initialize the similarity model ...')
        model = NearestNeighbors(metric=self.metric, n_neighbors=self.k+1, algorithm='auto')
        model.fit(self.ratings_matrix)
        return model

    def prepare_ratings(self):
        print('Normalize users ratings ...')
        print('Warning: make sure to name users column as userid, rating column as rating, item column as itemid.')
        means = self.ratings.groupby(by='userid', as_index=False)['rating'].mean()
        means_ratings = pd.merge(self.ratings, means, suffixes=('', '_mean'), on='userid')
        means_ratings['norm_rating'] = means_ratings['rating'] - means_ratings['rating_mean']

        return means.to_numpy()[:, 1], means_ratings

    def get_user_nearest_neighbors(self, userid):
        return self.similarities[userid], self.neighbors[userid]

    def compute_nearest_neighbors(self):
        print('Compute nearest neighbors ...')
        similarities, neighbors = self.model.kneighbors(self.ratings_matrix)
        return similarities[:, 1:], neighbors[:, 1:]

    def user_rated_items(self, userid):
        act = self.np_ratings[self.np_ratings[:, 0] == userid]
        items = act[:, 1]
        return items

    def find_user_candidate_items(self, userid, n=50):
        user_neighbors = self.neighbors[userid]
        user_rated_items = self.user_rated_items(userid)

        neighbors_rated_items = self.ratings.loc[self.ratings.userid.isin(user_neighbors)]
        items_frequencies = neighbors_rated_items.groupby('itemid')['rating']\
            .count()\
            .reset_index(name='count')\
            .sort_values(['count'], ascending=False)

        neighbors_rated_items_sorted_by_frequency = items_frequencies.itemid
        candidates_items = np.setdiff1d(neighbors_rated_items_sorted_by_frequency, user_rated_items, assume_unique=True)

        return candidates_items[:n]

    def similar_users_who_rated_this_item(self, userid, itemid):
        users_who_rated_this_item = self.np_ratings[self.np_ratings[:, 1] == itemid][:, 0]
        sim_users_who_rated_this_item = users_who_rated_this_item[np.isin(users_who_rated_this_item, self.neighbors[userid])]
        return users_who_rated_this_item, sim_users_who_rated_this_item

    def predict(self, userid, itemid):
        user_mean = self.means[userid]
        user_similarities = self.similarities[userid]
        user_neighbors = self.neighbors[userid]

        iratings = self.np_ratings[self.np_ratings[:, 1].astype('int') == itemid]

        suri = iratings[np.isin(iratings[:, 0], user_neighbors)]

        normalized_ratings = suri[:, 4]
        indexes = [np.where(user_neighbors == uid)[0][0] for uid in suri[:, 0].astype('int')]
        sims = user_similarities[indexes]

        num = np.dot(normalized_ratings, sims)
        den = np.sum(np.abs(sims))

        if num == 0 or den == 0:
            return user_mean

        r_hat = user_mean + np.dot(normalized_ratings, sims) / np.sum(np.abs(sims))

        return r_hat

    def evaluate(self, x_test, y_test):
        print('Evaluate the model on {} test data ...'.format(x_test.shape[0]))
        preds = list(self.predict(u, i) for (u, i) in x_test)
        mae = np.sum(np.absolute(y_test - np.array(preds))) / x_test.shape[0]
        print('MAE :', mae)
        return mae

    def user_predictions(self, userid, predictions_file):
        candidates = self.find_user_candidate_items(userid, n=30)
        for itemid in candidates:

            r_hat = self.predict(userid, itemid)

            with open(predictions_file, 'a+') as file:
                line = f'{userid},{itemid},{r_hat}\n'
                file.write(line)

    def all_predictions(self):
        users = self.ratings.userid.unique()

        now = str(datetime.now()).replace(' ', '-').split('.')[0]
        file_name = f'prediction.{now}.csv'
        predictions_file = os.path.join(self.predictions_dir, file_name)

        for userid in users:
            self.user_predictions(userid, predictions_file)

    def make_recommendations(self, userid):

        uid = self.uencoder.transform([userid])[0]
        predictions_files = glob(f'{self.predictions_dir}/*.csv')
        last_predictions = sorted(
            predictions_files, 
            key=lambda file: datetime.fromtimestamp(os.path.getmtime(file)),
            reverse=True
        )[0]

        predictions = pd.read_csv(
            last_predictions, sep=',', 
            names=['userid', 'itemid', 'predicted_rating']
        )
        predictions = predictions[predictions.userid == uid]
        recommendation_list = predictions.sort_values(
            by=['predicted_rating'], 
            ascending=False
        )

        recommendation_list.userid = self.uencoder.inverse_transform(recommendation_list.userid.tolist())
        recommendation_list.itemid = self.iencoder.inverse_transform(recommendation_list.itemid.tolist())

        recommendation_list = pd.merge(
            recommendation_list, 
            self.movies, 
            on='itemid', 
            how='inner'
        )

        return recommendation_list