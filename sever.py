from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Podcast Streaming Server")

# Temporary directory for audio files
TEMP_BASE = "/tmp"

# Request and response models for the chatbot
class ChatRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    response: str

@app.post("/generate", response_model=ChatResponse)
async def generate_response(request: ChatRequest):
    """
    Generate a response for the given prompt.
    """
    # Placeholder logic for generating a response
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    # Example response (replace with actual AI logic)
    ai_response = f"Echo: {request.prompt}"
    return ChatResponse(response=ai_response)