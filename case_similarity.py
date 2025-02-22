from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load dataset
df = pd.read_csv("7k  Unique crime articles.csv")

# Combine heading and content summary
df['combined_text'] = df['heading'].fillna('') + ' ' + df['content_summary'].fillna('')

# Create TF-IDF matrix
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_text'])

# Fit KNN model with cosine similarity
knn_model = NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute')
knn_model.fit(tfidf_matrix)

# Function to get similar articles for a given text input
def get_similar_articles_from_text(input_text, top_n=5):
    input_vector = tfidf_vectorizer.transform([input_text])  # Convert input text to TF-IDF vector
    distances, indices = knn_model.kneighbors(input_vector, n_neighbors=top_n + 1)
    
    similar_articles_indices = indices.flatten()[1:]  # Exclude the input itself
    return df.iloc[similar_articles_indices][['heading', 'content_summary', 'article_link']].to_dict(orient='records')

# Flask API endpoint to get similar articles based on text input
@app.route('/get_similar_articles', methods=['POST'])
def fetch_similar_articles():
    try:
        data = request.get_json()
        input_text = data.get("text", "")

        if not input_text.strip():
            return jsonify({"error": "Input text cannot be empty"}), 400

        top_n = int(data.get("top_n", 5))
        similar_articles = get_similar_articles_from_text(input_text, top_n)
        return jsonify({"similar_articles": similar_articles})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
