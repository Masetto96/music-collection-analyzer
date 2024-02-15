import os
import json
import pickle

import jsonlines

import numpy as np
import pandas as pd

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


def read_pickle_descriptors(pickle_file_path: str) -> pd.DataFrame:
    with open(pickle_file_path, 'rb') as f:
        data = pickle.load(f)
    df = pd.DataFrame(data, columns=['file_path', 'features'])
    # Unpack the dictionary of features
    df_features = pd.json_normalize(df['features'])
    # Combine the unpacked features DataFrame with the original DataFrame
    df = pd.concat([df[['file_path']], df_features], axis=1)
    return df


def load_essentia_analysis(ESSENTIA_ANALYSIS_PATH):
    return pd.read_pickle(ESSENTIA_ANALYSIS_PATH)

def read_numpy_arrays_from_pickle(filename):
    arrays = []
    try:
        with open(filename, 'rb') as file:
            while True:
                try:
                    array = pickle.load(file)
                    arrays.append(array)
                except EOFError:
                    break
    except Exception as e:
        print(f"An error occurred while reading: {e}")
    return np.array(arrays)
