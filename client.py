import requests

# URL of the FastAPI server
BASE_URL = "http://127.0.0.1:8000"

def test_generate_endpoint():
    """
    Test the /generate endpoint of the FastAPI server.
    """
    endpoint = f"{BASE_URL}/generate"
    payload = {"prompt": "What do I do if I touched poison ivy?"}

    try:
        response = requests.post(endpoint, json=payload)
        if response.status_code == 200:
            print("Response:", response.json())
        else:
            print(f"Error {response.status_code}: {response.json()}")
    except Exception as e:
        print("An error occurred:", str(e))

if __name__ == "__main__":
    test_generate_endpoint()