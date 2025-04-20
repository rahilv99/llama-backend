import sqlite3
import json

def query_embedding(word):
    # Connect to the database
    conn = sqlite3.connect("embeddings.db")
    cursor = conn.cursor()
    
    # Query for the specific word
    cursor.execute("SELECT vector FROM embeddings WHERE word = ?", (word,))
    result = cursor.fetchone()
    
    # Close connection
    conn.close()
    
    if result:
        # Parse the vector from JSON string back to list
        vector = json.loads(result[0])
        return vector
    else:
        return None

if __name__ == "__main__":
    # Query for "trail"
    word = "trail"
    vector = query_embedding(word)
    
    if vector:
        print(f"Vector for '{word}':")
        # Print the first 5 elements for brevity, then the length
        print(f"First 5 elements: {vector[:5]}")
        print(f"Total length: {len(vector)}")
    else:
        print(f"No embedding found for '{word}'") 