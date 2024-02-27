import os
import json
import pickle

import numpy as np

def load_json(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print("File not found.")
    except json.JSONDecodeError:
        print("Invalid JSON format.")

def read_numpy_arrays_from_pickle(filename, with_averaging=False):
    """
    Read numpy arrays from a pickle file.

    Args:
        filename (str): The name of the pickle file to read from.
        with_averaging (bool, optional): Whether to calculate the mean of the arrays. Defaults to False.

    Returns:
        list: A list of numpy arrays read from the pickle file.
    """
    arrays = []
    try:
        with open(filename, 'rb') as file:
            while True:
                try:
                    array = pickle.load(file)

                    if with_averaging:
                        array = np.mean(array, axis=0)

                    arrays.append(array)
                except EOFError:
                    break
    except Exception as e:
        print(f"An error occurred while reading: {e}")
    return arrays

def get_most_similar_embeddings(similarity_matrix: np.array, idx:int, k:int=10): # type: ignore
    """
    A function to get the most similar embeddings based on a similarity matrix.

    Parameters:
    similarity_matrix (np.array): The similarity matrix of embeddings.
    idx (int): The index of the vector of interest.
    k (int, optional): The number of most similar embeddings to return. Defaults to 10.

    Returns:
    np.array: An array of indexes representing the most similar embeddings.
    """
    # get the vector of interest, associated with the idx   
    # sort based on similarity
    # flip the order of the list to get most similar first, descending order
    most_similar_indexes = np.argsort(similarity_matrix[idx])[::-1]
    # exclude first element as it is the most similar, ofc it is the most similar to itself! ahah
    return most_similar_indexes[1:k+1] 



