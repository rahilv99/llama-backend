from typing import Dict
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
from geopy.geocoders import Nominatim

load_dotenv() 

oai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()
app = FastAPI(title="Hiking Llama Server")
model_to_train = "gpt-4.1-mini"
openai.api_key = oai_key

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
    dist_traveled: str
    dist_left: str
    days_traveled: str
    longitude: str
    latitude: str
    
class ContextRequest(BaseModel):
    url: str
    
class ContextResponse(BaseModel):
    vector_embeddings: list[list[float]]
    cleaned_info: str
    title: str

class WordResponse(BaseModel):
    vector_embeddings: Dict[str, list[float]]

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
    
    if not request.text.strip() and not request.image_base64.strip() and not (request.audio_base64 and request.audio_base64.strip()):
        raise HTTPException(status_code=400, detail="Need at least one source of information")
    inputs = []
    if request.text:
        inputs.append({"type": "input_text", "text": f"Here is the context: {request.text}. {request.prompt}" })
    if request.image_base64:
        inputs.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{request.image_base64}", "detail": "low", })
    if request.audio_base64:
        filename = "audio.m4a"
        decode_base64_to_mp3(request.audio_base64, filename)
        with open(filename, "rb") as f:
            transcriptions = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=f,
            )
            inputs.append({"type": "input_text", "text": f"This is a transcript of an audio file {transcriptions}" })

    response = client.responses.create(
    model=model_to_train,
    input=[{
        "role": "user",
        "content": inputs
        }],
    text={
            "format": {
                "type": "json_schema",
                "name": "steps-to-survive",
                "schema": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "steps": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["summary", "steps"],
                    "additionalProperties": False
                },
                "strict": True,
                #summary
                #array of strings for steps on what to do
            }
        }
    )
    return ChatResponse(response=response.output_text)

def decode_base64_to_mp3(base64_string: str, output_filename: str):
    try:
        m4a_bytes = base64.b64decode(base64_string)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid base64 audio") from e

    with open(output_filename, "wb") as f:
        f.write(m4a_bytes)
        
@app.post("/path", response_model=ChatResponse)
async def path(request: PathRequest):
    geo = Nominatim(user_agent="geocoding_app")
    coords = f"{request.latitude}, {request.longitude}"
    location = geo.reverse(coords)
    
    
    url = f"https://wttr.in/{request.latitude},{request.longitude}?format=j1"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()['current_condition'][0]
        temp = data["temp_C"]
        weather = data["weatherDesc"][0]["value"]
    else:
        weather = "unknown"
        temp = "unknown"        
    print(temp, weather)
        
    prompt = {"type": "input_text", "text": f"I am currently hiking at {location}. I have hiked {request.dist_traveled} miles in {request.days_traveled} days. I have {request.dist_left} miles left. The weather is currently {weather} and the temperature is {temp} celsius. Give suggestions on what to be out to look for based on the location, what I should prepare for the day, how much I should hike the next few days, and other relevant tips for the hike." }
    response = client.responses.create(
    model=model_to_train,
    input=[{
        "role": "user",
        "content": [prompt]
        }],
    text={
            "format": {
                "type": "json_schema",
                "name": "path-estimation",
                "schema": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "tips": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["summary", "tips"],
                    "additionalProperties": False
                },
                "strict": True,
                #summary
                #array of strings for steps on what to do
            }
        }
    )
    return ChatResponse(response=response.output_text)


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

@app.post("/context/words", response_model=WordResponse)
def vectorize_hiking_words():
    hiking_words = [
    "trail", "backpack", "boots", "summit", "peak", "ridge", "campsite", "tent",
    "compass", "map", "navigation", "trek", "expedition", "journey", "ascent",
    "descent", "altitude", "elevation", "path", "route", "waypoint", "landmark",
    "wilderness", "forest", "mountain", "hill", "valley", "creek", "river",
    "lake", "pond", "meadow", "glacier", "snowfield", "scree", "boulder",
    "rock", "canyon", "gorge", "waterfall", "stream", "bridge", "crossing",
    "switchback", "overlook", "vista", "scenic", "panorama", "binoculars",
    "daypack", "hydration", "canteen", "flask", "headlamp", "flashlight",
    "sunscreen", "bugspray", "trekking", "poles", "gaiters", "windbreaker",
    "raincoat", "poncho", "firstaid", "bandage", "blister", "carabiner",
    "harness", "helmet", "rope", "anchor", "belay", "scramble", "climb",
    "abseil", "rappel", "firestarter", "matches", "lighter", "knife",
    "multitool", "whistle", "beacon", "locator", "signal", "rescue",
    "survival", "shelter", "emergency", "trailhead", "parkinglot", "permit",
    "pass", "registration", "ranger", "station", "signpost", "marker",
    "blaze", "loop", "outandback", "thruhike", "sectionhike", "backcountry",
    "wildernessarea", "nationalpark", "statepark", "reserve", "conservation",
    "wildlife", "bear", "moose", "deer", "elk", "coyote", "fox", "mountainlion",
    "snake", "lizard", "eagle", "hawk", "owl", "birdwatching", "flora",
    "fauna", "wildflowers", "moss", "lichen", "fungi", "mushroom",
    "roots", "branches", "canopy", "underbrush", "thicket", "clearing",
    "trailmix", "granola", "energybar", "snack", "meal", "freeze-dried",
    "ration", "cookset", "stove", "fuel", "wood", "campfire", "firepit",
    "latrine", "outhouse", "wastebag", "leave-no-trace", "packout",
    "gear", "equipment", "supplies", "layers", "base-layer", "insulation",
    "outer-layer", "gloves", "beanie", "hat", "sunglasses", "buff",
    "scarf", "gaiter", "fleece", "downjacket", "softshell", "hardshell",
    "scrambling", "ridgewalk", "forestbath", "mossyrocks", "sunrisehike",
    "sunsetview", "trailrunner", "alpinezone", "barrenlands", "crag",
    "bushwhack", "lichenfield", "hiddenfalls", "shadygrove", "stonearch",
    "treecanopy", "ridgecrest", "meanderingpath", "steeppass",
    "coldcreek", "serenepond", "lonetree", "wildmeadow", "boulderdash",
    "barefoottrail", "pinegrove", "whisperingpines", "shiftingweather",
    "howlingwind", "snowdrift", "crunchingleaves", "leaflitter",
    "fallenlogs", "animaltrack", "rockscramble", "frostflowers",
    "dewypath", "foggytrail", "overgrownpath", "hiddenlake",
    "majesticfalls", "basin", "bouldergarden", "wildernesszone",
    "windsweptpeak", "icyledge", "wildberries", "beartracks",
    "gamepath", "shelteredcove", "dustytrail", "ridgecamp",
    "trailjunction", "mistyforest", "undergrowth", "shiftingterrain",
    "snowcaps", "glacierfield", "sheercliff", "narrowledge",
    "fernvalley", "wildorchid", "stonybrook", "slipperyrock",
    "meadowlark", "hummingbird", "trailcompanion", "windchill",
    "rockyplateau", "pinetrail", "valleyview", "brushfield",
    "coldspring", "hikeleader", "backpackstrap", "waterstop",
    "campsitereview", "switchbackturn", "stormcloud", "breezepass",
    "summitsign", "woodenbridge", "grassyknoll", "wildtrail",
    "hiddenmeadow", "stonytrail", "logbridge", "trailcrossing",
    "snowypath", "dustyclimb", "frostedridge", "icepatch",
    "ravinepass", "floodedpath", "trailgap", "wanderingtrail",
    "sunbeams", "cracklingfire", "quietgrove", "mountaindawn",
    "eveningtrek", "earlystart", "packlist", "supplycache",
    "riverford", "nightcamp", "noontimebreak", "hightrail",
    "rockledge", "woodlandpath", "twistingtrail", "ravinewalk",
    "creekbed", "breezymeadow", "hikerslog", "gearcheck",
    "treelinecamp", "ridgecampfire", "fogbank", "wildcamp",
    "rockfield", "elevationgain", "stonepath", "marshcrossing",
    "beaverdam", "coldfront", "windstorm", "silentvalley",
    "boulderfield", "wildshelter", "thickbush", "snowmelt",
    "rockhopping", "wildtrailhead", "earlymist", "streamcrossing",
    "highdeserthike", "crestsummit", "rockterrace", "fastpack",
    "lightpacking", "trailbuddy", "muddyboots", "puddlejump",
    "hiddenfalls", "cragview", "overlookpoint", "highpass",
    "ridgeview", "thundermountain", "snowyforest", "stonefootpath",
    "mountainpath", "craggytrail", "openmeadow", "alpinecamp",
    "sunsetpeak", "lowlandtrail", "nighttrek", "forestedge",
    "pinefield", "glacierpeak", "deertrail", "hiddenoverlook",
    "snowbank", "peakbagger", "stormytrail", "rockface", "hilltop",
    "mistyhike", "starrynights", "rivertrail", "desertwalk",
    "woodensteps", "trailridge", "thickmist", "summitpush",
    "overgrowntrail", "quietstream", "ridgeoverlook", "canyonwalk",
    "trailheadsign", "sunrisepeak", "moondance", "glacialmelt",
    "campkitchen", "rockoverhang", "forestpath", "baretrail",
    "tallgrass", "wildridges", "birdcalls", "dawncamp", "thickforest",
    "icytrail", "shadedvalley", "summitcamp", "weatherwatch",
    "stormwatch", "nightwatch", "earlylight", "sunshinehike",
    "pinemist", "shelteredcamp", "meadowhike", "fastascent",
    "longtrail", "granitepeak", "breezyridge", "fogline",
    "nightfallhike", "ridgewalkway", "adventuretrail", "crossingstream",
    "narrowtrail", "loosegravel", "wildvista", "trailcross", "ridgeescape",
    "shadycamp", "latecamp", "firstlight", "highcamp", "lowcamp",
    "basinhike", "desertpath", "coolshade", "packadjust", "shoerelace",
    "blusterywind", "wildstep", "craggyridge", "snowyhill", "moonlittrail",
    "eveningcamp", "fadinglight", "riverbend", "edgeofthewoods",
    "nightpack", "packhaul", "ridgeascent", "eaglesoar", "streamtrail",
    "glacialpath", "icebridge", "logcrossing", "trailgapjump",
    "rockscrambler", "graveltrail", "foresthike", "eveningfog",
    "brighttrail", "crispair", "firstsnow", "autumnhike", "springtrail",
    "drytrail", "dustyfield", "openridge", "creekcrossing",
    "survivalskills", "knottying", "signalbuilding",
    "distressignals", "emergencypreparedness"
    ]
    model = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L12-v3')
    json = {}
    for word in hiking_words:
        json[word] = model.encode(word)
    return WordResponse(vector_embeddings=json)
    
    

def scrape_website(url: str) -> str:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    
    texts = soup.stripped_strings
    return " ".join(texts)

#get 1000 most important words related to hiking and embbed each word. return json mapping word to embedding
