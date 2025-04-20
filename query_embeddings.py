import sqlite3
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


def load_embeddings(table_name, db_name="embeddings.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    if table_name == "embeddings":
        cursor.execute("SELECT word, vector FROM embeddings")
        rows = cursor.fetchall()
        embeddings = []
        labels = []
        for word, vector_json in rows:
            vector = json.loads(vector_json)
            embeddings.append(vector)
            labels.append(word)
    elif table_name == "UrlEmbeddings":
        cursor.execute("SELECT id, vector, cleaned_info FROM UrlEmbeddings")
        rows = cursor.fetchall()
        embeddings = []
        labels = []
        cleaned_infos = []
        for id, vector_json, cleaned_info in rows:
            vector = json.loads(vector_json)
            embeddings.append(vector)
            labels.append(id)  # or title if you prefer
            cleaned_infos.append(cleaned_info)
    else:
        raise ValueError("Invalid table name")

    conn.close()

    if table_name == "UrlEmbeddings":
        return np.array(embeddings), labels, cleaned_infos
    else:
        return np.array(embeddings), labels

def find_top_5_similar(prompt):
    model = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L12-v3')
    embeddings = model.encode(prompt).reshape(1, -1)  
    url_vectors, url_labels, url_cleaned_infos = load_embeddings("UrlEmbeddings")

    similarities = cosine_similarity(url_vectors, embeddings).flatten()
    top_5_idx = np.argsort(similarities)[-5:][::-1]

    top_5_cleaned_infos = [url_cleaned_infos[idx] for idx in top_5_idx]

    print("\nTop 5 matching entries:")
    for idx in top_5_idx:
        print(f"Similarity: {similarities[idx]:.4f} - Info: {url_cleaned_infos[idx][:50]}...")

    return top_5_cleaned_infos


if __name__ == "__main__":
    top_infos = find_top_5_similar("I have a huge rash on my forearm and my skin is turning orange")
    print("Top 5 cleaned_info:")
    for info in top_infos:
        print("-", info)
        