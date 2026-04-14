from flask import Flask, render_template, request
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load Dataset (ONLY ONCE)

column_names = ['user_id','item_id','rating','timestamp']
df = pd.read_csv('ml-100k/u.data', sep='\t', names=column_names)

movie_names = pd.read_csv(
    'ml-100k/u.item',
    sep='|',
    encoding='ISO-8859-1',
    header=None
)

movie_names = movie_names[[0,1]]
movie_names.columns = ['item_id','title']

df = pd.merge(df, movie_names, on='item_id')

# Movie statistics
movie_stats = df.groupby('title').agg({
    'rating': ['count', 'mean']
}).round(2)

movie_stats.columns = ['num_viewers', 'avg_rating']

# User-Movie Matrix
moviemat = df.pivot_table(index='user_id', columns='title', values='rating')

# Recommendation Function , based on similarity of user ratings between movies

def predict_movies(movie_name):
    movie_user_ratings = moviemat[movie_name]
    similar_to_movie = moviemat.corrwith(movie_user_ratings)

    corr_movie = pd.DataFrame(similar_to_movie, columns=['correlation'])
    corr_movie.dropna(inplace=True)

    corr_movie = corr_movie.join(movie_stats['num_viewers'])

    predictions = corr_movie[corr_movie['num_viewers'] > 100] \
        .sort_values(['correlation','num_viewers'], ascending=[False, False]) \
        .head(5)

    return predictions.index.tolist()


# Routes

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = None

    available_movies = sorted(
        movie_stats[movie_stats['num_viewers'] > 100].index.tolist()
    )

    if request.method == "POST":
        movie_name = request.form.get("movie")

        if movie_name in available_movies:
            recommendations = predict_movies(movie_name)

    return render_template(
        "index.html",
        movies=available_movies,
        recommendations=recommendations
    )


if __name__ == "__main__":
    app.run(debug=True)
