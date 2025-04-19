from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama
from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv() 

oai_key = os.getenv("OPENAI_API_KEY")
OpenAI.api_key = oai_key
client = OpenAI()
app = FastAPI(title="Streaming Server")

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
    mp3_base64: str  
    

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
async def process_media(request: ChatRequest):
    """
        Takes in mp3, turn to text, and prompt model with something, return result
        takes in text does same thing
        takes in image as base64 supports image+text+audio with anythign
    """
    
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    if request.text.strip() and request.image_base64.strip() and request.mp3_base64.strip():
        raise HTTPException(status_code=400, detail="Need at least one source of information")
    
    response = client.responses.create(
    model="gpt-4.1",
    input= request.text
    )
    print(response.output_text)
    return ChatResponse(response=response.output_text)