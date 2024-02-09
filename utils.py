import jsonlines
import json
import os
import pickle
import numpy as np


def parse_discogs_genre_activations(activations: np.array):
    """
    Takes as input the activations for all the 400 genre.
    Uses metadata to associate each activations with the corresponding genre.
    Returns human readable information :)
    """

    # Open the JSON file
    with open("metadata/discogs-effnet-bs64-1.json") as json_file:
        data = json.load(json_file)

    # Access the "classes" key
    classes = data["classes"]

    assert len(classes) == len(activations)

    return dict(zip(classes, activations.tolist()))


def save_result(path, name, file, pickle_only=True):

    if not pickle_only:
        with jsonlines.open(os.path.join(path, name, ".json"), mode="w") as writer:
            writer.write_all(file)

    with open(os.path.join(path, name, ".pkl"), "wb") as f:
        pickle.dump(file, f)
