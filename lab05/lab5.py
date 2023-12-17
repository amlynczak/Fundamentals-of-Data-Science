import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

Y = np.loadtxt("each_movie_data_Y.csv", delimiter=",")
R = np.loadtxt("each_movie_data_R.csv", delimiter=",")

Y = np.transpose(Y)
R = np.transpose(R)

#print(Y)

my_ratings = np.zeros(1682)

my_ratings[1] = 4
my_ratings[49] = 5
my_ratings[55] = 5
my_ratings[70] = 3
my_ratings[819] = 4
my_ratings[901] = 4
my_ratings[1225] = 3
my_ratings[126] = 4
my_ratings[131] = 2
my_ratings[171] = 5
my_ratings[173] = 4
my_ratings[204] = 1
my_ratings[180] = 4
my_ratings[224] = 2
my_ratings[232] = 1
my_ratings[266] = 1
my_ratings[430] = 3
my_ratings[634] = 2
my_ratings[68] = 4
my_ratings[81] = 3
my_ratings[93] = 3

def user_based_filtering(Y, R, my_ratings, k=5):
    distances = []
    similarities = []
    for user_ratings in Y:  # Iterate over users
        distance = cosine(my_ratings, user_ratings)
        distances.append(distance)
        similarities.append(1 - distance)

    sorted_similarities = sorted(similarities, reverse=True)

    weighted_sum = np.zeros_like(my_ratings)
    for i in range(k):
        neighbor_ratings = Y[np.argmax(similarities)]
        weighted_sum += neighbor_ratings * sorted_similarities[i]

    recommendation = weighted_sum / np.sum(np.abs(sorted_similarities))

    return recommendation

recommendation = user_based_filtering(Y, R, my_ratings)

movies = np.genfromtxt("movie_ids.txt", delimiter=" ", dtype=str, usecols=(1,), skip_header=0, encoding='latin-1')

for i in range(1682):
    if recommendation[i] != 0:
        print(f"Recommended: Movie {i + 1}, Title: {movies[i]}")