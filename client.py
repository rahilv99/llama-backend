import requests
import base64

# URL of the FastAPI server
BASE_URL = "http://127.0.0.1:8000"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def test_generate_endpoint():
    """
    Test the /generate endpoint of the FastAPI server.
    """
    img_b64 = encode_image("example.jpg")
    audio_b64 = encode_image("creep.mp3")

    endpoint = f"{BASE_URL}/process"
    payload = {"prompt": "give a solution to the problems given the text, image, and audio",
               "text": "I have a huge rash on my forearm and my skin is turning orange",
               "image_base64": img_b64,
               "audio_base64": audio_b64
               }

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