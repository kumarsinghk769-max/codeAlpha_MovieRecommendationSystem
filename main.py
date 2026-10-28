"""CodeAlpha Machine Learning Project: Content-Based Movie Recommendation."""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_movie_data():
    """Small example catalogue included for offline use."""
    movies = [
        ("The Matrix", "science fiction action virtual reality robots"),
        ("Inception", "science fiction thriller dreams mind bending heist"),
        ("Interstellar", "science fiction space exploration family black holes"),
        ("The Martian", "science fiction space survival astronaut mars"),
        ("John Wick", "action thriller assassin revenge"),
        ("Toy Story", "animation comedy friendship toys family"),
        ("Finding Nemo", "animation adventure ocean fish family"),
        ("The Lion King", "animation adventure animals family musical"),
        ("The Notebook", "romance drama love relationship"),
        ("Pride and Prejudice", "romance drama love period adaptation"),
        ("The Conjuring", "horror supernatural haunted house thriller"),
        ("A Quiet Place", "horror thriller survival monsters family"),
        ("The Dark Knight", "action crime superhero thriller gotham"),
        ("Spider-Man: Into the Spider-Verse", "animation action superhero multiverse"),
        ("Coco", "animation family music adventure")
    ]
    return pd.DataFrame(movies, columns=["title", "description"])


def recommend_movies(title, movies, similarity, top_n=5):
    matches = movies.index[movies["title"].str.casefold() == title.casefold()].tolist()
    if not matches:
        options = ", ".join(movies["title"].tolist())
        raise ValueError(f"Movie not found. Choose one of these titles: {options}")

    selected_index = matches[0]
    ranked = sorted(
        enumerate(similarity[selected_index]),
        key=lambda item: item[1],
        reverse=True
    )
    return [
        (movies.iloc[index]["title"], float(score))
        for index, score in ranked if index != selected_index
    ][:top_n]


def main():
    movies = build_movie_data()
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(movies["description"])
    similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

    print("=== CodeAlpha Movie Recommendation System ===")
    print("\nAvailable movies:")
    for title in movies["title"]:
        print(f"- {title}")

    selected = input("\nEnter a movie title (press Enter for Inception): ").strip()
    if not selected:
        selected = "Inception"

    try:
        results = recommend_movies(selected, movies, similarity)
        print(f"\nRecommendations for '{selected}':")
        for rank, (title, score) in enumerate(results, start=1):
            print(f"{rank}. {title} (similarity: {score:.2f})")
    except ValueError as error:
        print(f"\n{error}")


if __name__ == "__main__":
    main()
