# CodeAlpha Movie Recommendation System

**Machine Learning Project | Content-Based Filtering**

## Objective
Recommend movies similar to a movie selected by the user.

## How it works
1. Load a small movie catalogue with titles and descriptions.
2. Use `TfidfVectorizer` to convert descriptions into numerical vectors.
3. Calculate similarity using cosine similarity.
4. Display the five most similar movies, ranked by similarity score.

## Requirements
- Python 3.9+
- pandas
- scikit-learn

## Run locally
```bash
python -m pip install -r requirements.txt
python main.py
```

When asked, enter a title from the displayed list. Press Enter to use `Inception`.

## Project files
- `main.py` — movie catalogue, text processing, similarity calculation, and recommendations.
- `requirements.txt` — Python dependencies.

## Dataset
A small example catalogue is included in the code, so no download is needed.

## Limitations
This is a beginner demonstration, not a commercial recommendation engine. It uses movie descriptions only; it does not use user ratings, viewing history, or a large real-world dataset. Similarity scores represent text similarity and do not guarantee that a user will like a movie.
