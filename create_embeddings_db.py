import json
import sqlite3
import requests
import os

BASE_URL = "http://127.0.0.1:8000"

def create_url_embeddings_database(url):
    endpoint = f"{BASE_URL}/context"
    payload = {"url": url}

    try:
        response = requests.post(endpoint, json=payload)
        if response.status_code == 200:
            data = response.json()
            print("Response:", response.json())
        else:
            print(f"Error {response.status_code}: {response.json()}")
    except Exception as e:
        print("An error occurred:", str(e))
    conn = sqlite3.connect("embeddings.db")
    cursor = conn.cursor()  
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS UrlEmbeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        vector TEXT NOT NULL,
        cleaned_info TEXT,
        title TEXT
    )
    ''')
    try:
        cursor.execute(
            "INSERT OR REPLACE INTO UrlEmbeddings (vector, cleaned_info, title) VALUES (?, ?, ?)",
            (json.dumps(data['vector_embeddings'][0]), data['cleaned_info'], data['title'])
        )
        
        # Commit changes and close connection
        conn.commit()
        print(f"Successfully added url embedding to embeddings.db")
    except Exception as e:
        print(f"Error populating database: {e}")
    finally:
        conn.close()
      
def create_embeddings_database():
    # Check if word_response.json exists
    if not os.path.exists("word_response.json"):
        print("Error: word_response.json not found. Run test_word_endpoint() first.")
        return
    
    # Load JSON data
    try:
        with open("word_response.json", "r") as file:
            data = json.load(file)
    except json.JSONDecodeError:
        print("Error: Invalid JSON data in word_response.json")
        return
    
    # Check for the expected structure
    if "vector_embeddings" not in data:
        print("Error: Expected 'vector_embeddings' key not found in JSON")
        return
    
    # Connect to SQLite database
    conn = sqlite3.connect("embeddings.db")
    cursor = conn.cursor()
    
    # Create table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS embeddings (
        word TEXT PRIMARY KEY,
        vector TEXT
    )
    ''')
    
    # Extract word-vector pairs and insert into database
    try:
        count = 0
        for word, vector in data["vector_embeddings"].items():
            # Convert vector to JSON string for storage
            vector_json = json.dumps(vector)
            cursor.execute(
                "INSERT OR REPLACE INTO embeddings (word, vector) VALUES (?, ?)",
                (word, vector_json)
            )
            count += 1
        
        # Commit changes and close connection
        conn.commit()
        print(f"Successfully added {count} words to embeddings.db")
    except Exception as e:
        print(f"Error populating database: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    #create_embeddings_database() 
    create_url_embeddings_database("https://koa.com/blog/hiking-tips-and-tricks-how-to-plan-and-prepare-for-a-hike/")