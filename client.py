import requests
import base64
import json

# URL of the FastAPI server
BASE_URL = "http://127.0.0.1:8000"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def test_process_endpoint():
    img_b64 = encode_image("rash.jpg")
    #audio_b64 = encode_image("creep.mp3")

    endpoint = f"{BASE_URL}/process"
    payload = {"prompt": "give a solution to the problems given the text, image, and audio",
               "text": "I have a huge rash on my forearm and my skin is turning orange",
               "image_base64": img_b64,
               "audio_base64": ""
               }

    try:
        response = requests.post(endpoint, json=payload)
        if response.status_code == 200:
            print("Response:", response.json())
        else:
            print(f"Error {response.status_code}: {response.json()}")
    except Exception as e:
        print("An error occurred:", str(e))
        
def test_path_endpoint():
    endpoint = f"{BASE_URL}/path"
    payload = {
        "dist_traveled": "10 miles",
        "dist_left": "30 miles",
        "days_traveled": "1",
        "latitude": "45.8326",
        "longitude": "6.8652"
    }
    try:
        response = requests.post(endpoint, json=payload)
        if response.status_code == 200:
            print("Response:", response.json())
        else:
            print(f"Error {response.status_code}: {response.json()}")
    except Exception as e:
        print("An error occurred:", str(e))

def test_context_endpoint():
    endpoint = f"{BASE_URL}/context"
    payload = {"url": "https://www.thenightsky.com/blog/identify-stars-for-stargazing"}

    try:
        response = requests.post(endpoint, json=payload)
        if response.status_code == 200:
            print("Response:", response.json())
        else:
            print(f"Error {response.status_code}: {response.json()}")
    except Exception as e:
        print("An error occurred:", str(e))
        
def test_word_endpoint():
    endpoint = f"{BASE_URL}/context/words"
    payload = {}
    try:
        response = requests.post(endpoint, json=payload)
        if response.status_code == 200:
            # Write response to a JSON file
            with open("word_response.json", "w") as file:
                json.dump(response.json(), file, indent=4)
            print("Response saved to word_response.json")
        else:
            print(f"Error {response.status_code}: {response.json()}")
    except Exception as e:
        print("An error occurred:", str(e))

if __name__ == "__main__":
    #test_process_endpoint()
    #test_context_endpoint()
    #test_word_endpoint()
    test_path_endpoint()