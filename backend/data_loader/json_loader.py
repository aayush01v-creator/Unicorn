import json

def load_network(path: str):
    with open(path, "r") as f:
        return json.load(f)
