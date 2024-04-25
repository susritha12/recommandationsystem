import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

ratings_data = pd.read_csv('ratings.csv')
movies_data = pd.read_csv('movies.csv')


train_data, test_data = train_test_split(ratings_data, test_size=0.2, random_state=42)


train_matrix = train_data.pivot_table(index='userId', columns='movieId', values='rating', fill_value=0)
test_matrix = test_data.pivot_table(index='userId', columns='movieId', values='rating', fill_value=0)


user_similarity = train_matrix.corr(method='pearson')


def predict(ratings, similarity):
    mean_user_rating = ratings.mean(axis=1)
    ratings_diff = (ratings - mean_user_rating[:, None])
    pred = mean_user_rating[:, None] + similarity.dot(ratings_diff) / similarity.sum(axis=1)[:, None]
    return pred


train_pred = predict(train_matrix.values, user_similarity.values)
test_pred = predict(test_matrix.values, user_similarity.values)


def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))


train_rmse = rmse(train_pred, train_matrix.values)
test_rmse = rmse(test_pred, test_matrix.values)

print(f"Train RMSE: {train_rmse}")
print(f"Test RMSE: {test_rmse}")
