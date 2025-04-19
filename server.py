from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_cpp import Llama
import requests
import openai
from openai import OpenAI
from dotenv import load_dotenv
import base64
import os
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup

load_dotenv() 

oai_key = os.getenv("OPENAI_API_KEY")
openai.api_key = oai_key
client = OpenAI()
app = FastAPI(title="Streaming Server")
model_to_train = "gpt-4.1-mini"

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MOCK = False

# Request and response models for the chatbot
class ChatRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    response: str
    
class ProcessRequest(BaseModel):
    prompt: str
    text: str
    image_base64: str 
    audio_base64: str    
    
class PathRequest(BaseModel):
    curr_loc: str
    dest: str
    start: str
    
class PathResponse(BaseModel):
    path: list[str]
    
class ContextRequest(BaseModel):
    url: str
    
class ContextResponse(BaseModel):
    vector_embeddings: list[list[float]]
    cleaned_info: str
    title: str

llm = Llama.from_pretrained(
	repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
	filename="Llama-3.2-3B-Instruct-Q5_K_S.gguf",
)

@app.post("/generate", response_model=ChatResponse)
async def generate_response(request: ChatRequest):
    """
    Generate a response for the given prompt.
    """
    # Placeholder logic for generating a response
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    # Example response (replace with actual AI logic)
    if MOCK:
        ai_response = f"Echo: {request.prompt}"
    else:
        out = llm.create_chat_completion(
            messages = [
                {
                    "role": "system",
                    "content": "You are an AI companion for a user in a remote situation. They may need immeadiate care and assistance. \
                        Provide useful instruction in a concise format."
                },
                {
                    "role": "user",
                    "content": request.prompt
                }
            ]
        )

        # Extract the AI's response from the output
        ai_response = out["choices"][0]["message"]["content"]

    return ChatResponse(response=ai_response)

@app.post("/process", response_model=ChatResponse)
async def process_media(request: ProcessRequest):
    """
        Takes in mp3, turn to text, and prompt model with something, return result
        takes in text does same thing
        takes in image as base64 supports image+text+audio with anything
    """
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty") 
    
    if not request.text.strip() and not request.image_base64.strip() and request.audio_base64.strip() is None:
        raise HTTPException(status_code=400, detail="Need at least one source of information")
    inputs = []
    if request.text:
        inputs.append({"type": "input_text", "text": f"Here is the context: {request.text}. {request.prompt}" })
    if request.image_base64:
        inputs.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{request.image_base64}", "detail": "low", })
    if request.audio_base64:
        filename = "audio.mp3"
        decode_base64_to_mp3(request.audio_base64, filename)
        transcriptions = client.audio.transcriptions.create(
        model="gpt-4o-transcribe",
        file=open("audio.mp3", "rb"),
        )
        inputs.append({"type": "input_text", "text": f"This is a transcript of an audio file {transcriptions}" })

    response = client.responses.create(
    model=model_to_train,
    input=[{
        "role": "user",
        "content": inputs
        }]
    )
    return ChatResponse(response=response.output_text)

def decode_base64_to_mp3(base64_string: str, output_filename: str):
    # Step 1: Decode base64 string to bytes
    mp3_bytes = base64.b64decode(base64_string)
    with open(output_filename, "wb") as f:
        f.write(mp3_bytes)
        
@app.post("/compute_paths", response_model=ChatResponse)
async def path_computation(request: PathRequest): 
    pass

@app.post("/context", response_model=ContextResponse)
async def get_context_data(request: ContextRequest): 
    """
        Given a websites, scrape the data, input into LLM to clean the data, vectorize it, and send the vectors and summary pair
    """
    text = scrape_website(request.url)
    #print(f"[{request.url}] {text}")
    response =  client.responses.create(
            model=model_to_train,
            input=[{
                "role": "user",
                "content": f"Clean this content for only relevant information and provide the title {text}. Format it into cleaned_info: and title:"
                }]
            )
    lines = response.output_text.strip().splitlines()
    cleaned_info = ""
    title = lines[0][len("title: "):].strip()
    for line in range(3, len(lines)):
        cleaned_info = "".join(lines[line])
        
    model = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L12-v3')
    embeddings = model.encode(lines[3: len(lines)])
    return ContextResponse(vector_embeddings=embeddings.tolist(), cleaned_info=cleaned_info, title=title)

def scrape_website(url: str) -> str:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    
    texts = soup.stripped_strings
    return " ".join(texts)

#get 1000 most important words related to hiking and embbed each word. return json mapping word to embedding