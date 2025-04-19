# llama-backend


## Quickstart
1. Clone repo
2. Run `python -m venv venv` to create a virtual environment
2. Run `source venv/bin/activate` in a bash window
3. Install dependenices with `pip install -r requirements.txt`
4. Make sure uvicorn is installed. For linux: `sudo apt install uvicorn`
4. Launch server to port 8000 `uvicorn sever:app --reload`

## Testing
Open a new terminal
Run client.py with `python client.py`