# llama-backend


## Quickstart
1. Clone repo
2. Run `python -m venv venv` to create a virtual environment
2. Run `source venv/bin/activate` in a bash window
3. Install dependenices with `pip install -r requirements.txt`
4. Make sure uvicorn is installed. For linux: `sudo apt install uvicorn`
4. Launch server to port 8000 `uvicorn server:app --reload`
5. use an OPENAI key/ deepinfra key if you wish to use Llama 90B, make sure the model also matches the key you use

## Testing
Open a new terminal
Run client.py with `python client.py`

## Model hosting
1. Run `sudo apt install build-essential gcc-11 g++-11` to install gcc-11 c compiler
2. Export compiler variables
`export CC=gcc-11`
`export CXX=g++-11`
3. Run `pip install llama-cpp-python`