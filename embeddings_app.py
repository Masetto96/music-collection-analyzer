import os
import streamlit as st
import numpy as np
import helpers.utils as u
import helpers.streamlit_helpers as st_helpers
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDING_PATH = "embeddings"

with open(os.path.join(EMBEDDING_PATH, "embed_idxs.txt"), "r") as handle:
    idxs_path = handle.read().split("\n")[:-1]

# loading embeddings
discogs = u.read_numpy_arrays_from_pickle(os.path.join(EMBEDDING_PATH, "discogs.pkl"), with_averaging=True)
musiccnn = u.read_numpy_arrays_from_pickle(os.path.join(EMBEDDING_PATH, "musiccnn.pkl"), with_averaging=True)

discogs = np.array(discogs)
musiccnn = np.array(musiccnn)

# TITLE
st.title("Compare songs by cosine similarity!")
st.subheader(" 🎛️ This app uses musiccnn and discogs-effnet embeddings to compute music similarity across tracks 🎚️ ")

st.warning("Thouhts on the following track? :smile: ")
st.audio("audio/audio.000/1P/1P12MkjjBnaC26XnyVNZ3G.mp3", format="audio/mp3", start_time=0)
bad_song = idxs_path.index("audio/audio.000/1P/1P12MkjjBnaC26XnyVNZ3G.mp3")

similarity_matrix_discogs = cosine_similarity(discogs)
similarity_matrix_musiccnn = cosine_similarity(musiccnn)

st.write("## 🔘 Track Selection ")
selected_track = st.select_slider("The track you select will be used to compute similarity:", idxs_path, label_visibility="visible")
st.write("### 🎧 You have selected the following track, you can listen to it before proceeding ")
st.audio(selected_track, format="audio/mp3", start_time=0)

# Widget to input the number of tracks
num_tracks = st.number_input("How many tracks do you want to consider?", min_value=1, step=1, value=10, max_value=10)

# Display the selected number of tracks
st.write("Number of tracks selected:", num_tracks)

if st.button("# 🏃 RUN"):
    # get idx of the track that the user has selected
    track_idx = idxs_path.index(selected_track)

    st.write("You have selected", selected_track, track_idx)

    # calls get most similar from utils.py
    discogs_idxs = u.get_most_similar_embeddings(similarity_matrix=similarity_matrix_discogs, idx=track_idx, k=num_tracks)
    musiccnn_idxs = u.get_most_similar_embeddings(similarity_matrix=similarity_matrix_musiccnn, idx=track_idx, k=num_tracks)
    st.write("# 🔍 Results in asceding order of similarity")
    
    st.write("### 📜 Most similar tracks from DISCOGS-EFFNET &darr; ")
    st_helpers.display_audio_previews([idxs_path[i] for i in discogs_idxs])

    st.write("### 📜 Most similar tracks from MUSICNN &darr; ")
    st_helpers.display_audio_previews([idxs_path[i] for i in musiccnn_idxs])



