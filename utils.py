import jsonlines
import json
import os
import pickle

def load_json(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print("File not found.")
    except json.JSONDecodeError:
        print("Invalid JSON format.")

def save_result(path, name, file, pickle_only=True):
    if not pickle_only:
        with jsonlines.open(os.path.join(path, name + ".json"), mode="w") as writer:
            writer.write_all(file)

    with open(os.path.join(path, name + ".pkl"), "wb") as f:
        pickle.dump(file, f)