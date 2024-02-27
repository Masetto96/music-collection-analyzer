import os

import pandas as pd
import streamlit as st
import helpers.utils as u
import helpers.playlist_options as playlist_helpers
import helpers.streamlit_helpers as st_helpers

# ESSENTIA_ANALYSIS_PATH = 'data/files_essentia_effnet-discogs.jsonl.pickle'
DISCOGS_400_GENRE_ACTIVATIONS_PATH = "descriptors/400-genre-activations.pkl"
DESCRIPTORS_PATH = "descriptors/descriptors-but-genre-v4.pkl"
PLAYLIST_PATH = "playlists"

genre_activations = pd.DataFrame(pd.read_pickle(DISCOGS_400_GENRE_ACTIVATIONS_PATH))
essentia_descriptors = pd.DataFrame(pd.read_pickle(DESCRIPTORS_PATH))


## DISCLAIMER:
# The following line is needed for my implementations as I run the extractor on different machines and the structure of the directories was not the same (in 1 case it was nested)
# it simply removes the prefix /audio from the file_path column
genre_activations["file_path"] = genre_activations["file_path"].apply(
    lambda x: x.removeprefix("audio/")
)

# # sort the dfs according to file path columns so idxs are consistent
genre_activations = genre_activations.sort_values(by="file_path").reset_index(drop=True)
essentia_descriptors = essentia_descriptors.sort_values(by="file_path").reset_index(drop=True)


# TITLE
st.title(" 💿 Playlist Builder")
st.subheader("Choose the method you want to use to create a playlist from the dropdown list on the left side of the screen.")
st.write(
    f"Using analysis data, extracted with https://essentia.upf.edu/, from `{DISCOGS_400_GENRE_ACTIVATIONS_PATH}` and `{DESCRIPTORS_PATH}`."
)

st.sidebar.header("Playlist Creation")
playlist_option = st.sidebar.selectbox(
    "Create playlist based on:",
    ["Genre", "BPM", "Arousal-Valence", "Danceability", "Key-Scale", "Vocal-Instrumental"],
)
st.write("Loaded audio analysis for", len(genre_activations), "tracks.")
audio_analysis_styles = genre_activations.columns[1:]

# FILTERS SELECTION
st.write("## 🔍 Select")
if playlist_option == "Genre":
    st.write("### By style")
    st.write("Style activation statistics:")
    st.write(genre_activations.describe())
    st_helpers.plot_parent_genre_distribution(genre_activations)
    style_select = st.multiselect("Select by style activations:", audio_analysis_styles)
    if style_select:
        st.write(genre_activations[style_select].describe())
        style_select_str = ", ".join(style_select)
        style_select_range = st.slider(
            f"Select tracks with `{style_select_str}` activations within range:",
            value=[0.5, 1.0],
        )


if playlist_option == "BPM":
    min_tempo, max_tempo = playlist_helpers.select_by_bpm(essentia_descriptors)

if playlist_option == "Arousal-Valence":
    min_arousal, max_arousal, min_valence, max_valence = playlist_helpers.select_by_arousal_valence(essentia_descriptors)

if playlist_option == "Vocal-Instrumental":
    is_instrumental = playlist_helpers.select_vocal_instrumental(essentia_descriptors)

if playlist_option == "Danceability":
    lower_bound_dance, upper_bound_dance = playlist_helpers.select_by_danceability(essentia_descriptors)

if playlist_option == "Key-Scale":
    key, mode = playlist_helpers.select_by_key_scale(essentia_descriptors)


st.write("## 🔀 Post-process")

st.write("## 🔝 Rank")
style_rank = st.multiselect(
    "Rank by style activations (multiplies activations for selected styles):",
    audio_analysis_styles,
    [],
)

st.write('## 🧮')
max_tracks = st.number_input("Maximum number of tracks (0 for all):", value=0)

st.write('## 🎲 Shuffle ')
shuffle = st.checkbox("Random shuffle")

st.write('## 💾 Save')
save_file = st.checkbox("Save Playlist?")
if save_file:
    file_name = st.text_input("Enter the name for the playlist:")
    if os.path.exists(os.path.join(PLAYLIST_PATH, file_name + ".m3u8")):
        st.error("A file with this name already exists!")
        file_name = None

if st.button("RUN"):
    idx_to_path = essentia_descriptors["file_path"]
    mp3s_idxs = list(essentia_descriptors.index)
    st.write("## 🔊 Results")

    if playlist_option == "Genre":
        audio_analysis_query = genre_activations.loc[mp3s_idxs][style_select]
        result = audio_analysis_query
        for style in style_select:
            result = result.loc[result[style] >= style_select_range[0]]
        st.write(result)
        mp3s_idxs = result.index
        st.success(f"Found {len(mp3s_idxs)} songs")

    if playlist_option == "BPM":
        mp3s_idxs = st_helpers.filter_by_bpm(
            essentia_descriptors, max_bpm=max_tempo, min_bpm=min_tempo
        )
        st.success(f"Found {len(mp3s_idxs)} songs")
        # st.dataframe(essentia_descriptors.iloc[mp3s_idxs]["tempo"])s

    elif playlist_option == "Danceability":
        mp3s_idxs = st_helpers.filter_by_danceability(
            essentia_descriptors,
            danceability_lower=lower_bound_dance,
            danceability_upper=upper_bound_dance,
        )
        st.success(f"Found {len(mp3s_idxs)} songs")

    elif playlist_option == "Arousal-Valence":
        mp3s_idxs = st_helpers.filter_by_arousal_and_valence(
            audio_analysis=essentia_descriptors,
            max_valence=max_valence,
            min_valence=min_valence,
            max_arousal=max_arousal,
            min_arousal=min_arousal,
        )
        st.success(f"Found {len(mp3s_idxs)} songs")
        # st.dataframe(essentia_descriptors.iloc[mp3s_idxs]["valence_arousal"])

    elif playlist_option == "Key-Scale":
        mp3s_idxs = st_helpers.filter_by_key_scale(essentia_descriptors, key, mode)
        st.success(f"Found {len(mp3s_idxs)} songs")
        # st.dataframe(essentia_descriptors.iloc[mp3s_idxs]["keyscale_edma"].apply(lambda x:(x[0],x[1])))

    elif playlist_option == "Vocal-Instrumental":
        mp3s_idxs = st_helpers.filter_vocal_instrumental(essentia_descriptors, is_instrumental)
        st.success(f"Found {len(mp3s_idxs)} songs")
        # st.dataframe(essentia_descriptors.iloc[mp3s_idxs]["voice_instrumental"])

    # POST PROCESSING
    if style_rank:
        ranked = st_helpers.rank_by_style(genre_activations, style_rank, mp3s_idxs)
        st.write("Applied ranking by audio style predictions.")
        st.write(ranked)
        mp3s_idxs = list(ranked.index)

    if max_tracks:
        mp3s_idxs = mp3s_idxs[:max_tracks]
        st.write("Using top", len(mp3s_idxs), "tracks from the results.")

    if shuffle:
        mp3s_idxs = st_helpers.shuffle_tracks(list(mp3s_idxs))
        st.write("Applied random shuffle.")

    if save_file and file_name:
        st_helpers.save_playlist(
            file_name,
            idx_to_path.iloc[mp3s_idxs].tolist(),
            PLAYLIST_PATH,
        )
        st.success(f'Stored M3U playlist (local filepaths) to `{file_name+".m3u8"}`.')
    else:
        st.warning("Playlist not saved.")

    # # getting a list of strings of the audio files that will be played
    mp3ss_audio_list = idx_to_path.iloc[mp3s_idxs].tolist()
    st_helpers.display_audio_previews(mp3ss_audio_list)
