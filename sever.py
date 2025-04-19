from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama

app = FastAPI(title="Streaming Server")

MOCK = False

# Request and response models for the chatbot
class ChatRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    response: str

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