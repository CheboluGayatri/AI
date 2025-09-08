# movie recommendation system using cosine similarity
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample dataset
data = {
    'movie_id': [1, 2, 3, 4, 5],
    'title': ['The Matrix', 'John Wick', 'Inception', 'Interstellar', 'The Dark Knight'],
    'genre': ['Action Sci-fi', 'Action Crime', 'Sci-fi Thriller', 'Sci-fi Drama', 'Action Thriller']
}

# Convert to DataFrame
df = pd.DataFrame(data)
print("Movie Data:")
print(df)

# TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['genre'])

print("TF-IDF Matrix shape:", tfidf_matrix.shape)

# Cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# function to get movie recommendations
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = df[df['title'] == title].index[0]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 5 most similar movies
    sim_scores = sim_scores[1:6]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 5 most similar movies
    return df['title'].iloc[movie_indices]

# Test the recommendation system
movie_title = 'Inception'
recommended_movies = get_recommendations(movie_title)
print(f"Movie recommendations for '{movie_title}':")
for movie in recommended_movies:
    print(movie)