from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load the dataset and preprocess
df = pd.read_csv('./books.csv', encoding='utf-8', on_bad_lines='skip')
df['title'] = df['title'].str.strip().str.lower()
df['authors'] = df['authors'].str.strip().str.lower()
df = df.drop_duplicates(subset=['title']).dropna(subset=['title', 'authors'])

def combine_features(row):
    return f"{row['title']} {row['authors']}"

df['combined_features'] = df.apply(combine_features, axis=1)
vectorizer = TfidfVectorizer(stop_words='english')
features = vectorizer.fit_transform(df['combined_features'])

# API Endpoint
@app.get("/recommend/{title}")
async def recommend_books(title: str):
    normalized_title = title.strip().lower()
    matches = df[df['title'].str.contains(normalized_title, na=False)]
    if matches.empty:
        return {"error": "Book not found in dataset."}

    book_index = matches.index[0]
    cosine_sim = cosine_similarity(features[book_index], features).flatten()
    similar_books = cosine_sim.argsort()[-6:-1][::-1]
    recommendations = df.iloc[similar_books][['title', 'authors']].to_dict(orient='records')
    return {"recommendations": recommendations}
    
