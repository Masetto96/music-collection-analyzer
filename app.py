"""
For voice/instrumental classifier, implement a binary selector that allows you to show only music with or without voice.

For danceability, implement a range selector. In the case of Essentia's signal processing algorithm, its output is within the [0, 3] range. In the case of the classifier, the output value is probability of music being danceable, within [0,1] range.

For arousal and valence, the range of values output by the model is within [1, 9]. Implement a range selector.

For key/scale, you can have a dropdown menu to select the key and select major vs minor scale. We have estimations done by three different profiles (`temperley`, `krumhansl`, or `edma`). Decide which of them you want to use for the UI.

"""

import os

import streamlit as st
import matplotlib.pyplot as plt
import utils as u
import streamlit_helpers as st_helpers
import seaborn as sns

# ESSENTIA_ANALYSIS_PATH = 'data/files_essentia_effnet-discogs.jsonl.pickle'
DISCOGS_400_GENRE_ACTIVATIONS_PATH = "descriptors/discogs-400-genre.pkl"
DESCRIPTORS_PATH = "descriptors/descriptors-but-genre.pkl"
PLAYLIST_PATH = "playlists"

# audio_analysis = u.load_DISCOGS_EFFNECT_GENRE(DISCOGS_GENRE_ACTIVATIONS_PATH)
genre_activations = u.read_pickle_descriptors(DISCOGS_400_GENRE_ACTIVATIONS_PATH)
essentia_descriptors = u.read_pickle_descriptors(DESCRIPTORS_PATH)

# TITLE
st.title("wip title")
st.write(
    f"Using analysis data from `{DISCOGS_400_GENRE_ACTIVATIONS_PATH}` and `{DESCRIPTORS_PATH}`."
)

st.sidebar.header("Playlist Creation")
playlist_option = st.sidebar.selectbox(
    "Create playlist based on:", ["Genre", "BPM", "Arousal-Valence"]
)
st.write("Loaded audio analysis for", len(genre_activations), "tracks.")
audio_analysis_styles = genre_activations.columns[1:]

# FILTERS SELECTION
st.write("## 🔍 Select")
if playlist_option == "Genre":
    st.write("### By style")
    st.write("Style activation statistics:")
    # first column is file_path
    st.write(genre_activations.describe())
    st.info("Genres available:")
    st.dataframe(audio_analysis_styles, use_container_width=True)
    style_select = st.multiselect("Select by style activations:", audio_analysis_styles)
    if style_select:
        st.write(genre_activations[style_select].describe())
        style_select_str = ", ".join(style_select)
        style_select_range = st.slider(
            f"Select tracks with `{style_select_str}` activations within range:",
            value=[0.5, 1.0],
        )

if playlist_option == "BPM":
    st.write("### By Beats per Minute")
    fig, ax = plt.subplots()
    ax.hist(essentia_descriptors["tempo"], color="skyblue", edgecolor="black", bins=20)
    ax.set_title("Distribution of BPM Data")
    ax.set_xlabel("Beats per minute")
    ax.set_ylabel("Frequency")
    ax.grid(True)

    st.pyplot(fig)
    min_tempo = st.number_input("Minimum Tempo (BPM):", value=60)
    max_tempo = st.number_input("Maximum Tempo (BPM):", value=200)

if playlist_option == "Arousal-Valence":
    joint_plot = sns.jointplot(
        x=essentia_descriptors["arousal_valence"].apply(lambda x: x[0]),
        y=essentia_descriptors["arousal_valence"].apply(lambda x: x[1]),
        kind="hex",
        color="skyblue",
    )
    joint_plot.set_axis_labels("Arousal", "Valence", fontsize=11)
    min_arousal, max_arousal = st.slider(
        "Select Arousal Range:", min_value=1, max_value=9, value=(1, 9)
    )
    min_valence, max_valence = st.slider(
        "Select Valence Range:", min_value=1, max_value=9, value=(1, 9)
    )
    st.pyplot(joint_plot)

    foo = st_helpers.filter_by_arousal_and_valence()


st.write("## 🔝 Rank")
style_rank = st.multiselect(
    "Rank by style activations (multiplies activations for selected styles):",
    audio_analysis_styles,
    [],
)

st.write("## 🔀 Post-process")
max_tracks = st.number_input("Maximum number of tracks (0 for all):", value=0)
shuffle = st.checkbox("Random shuffle")

save_file = st.checkbox("Save Playlist?")
if save_file:
    file_name = st.text_input("Enter the name for the playlist:")
    if os.path.exists(os.path.join(PLAYLIST_PATH, file_name + ".m3u8")):
        st.error("A file with this name already exists!")
        file_name = None

if st.button("RUN"):
    st.write("## 🔊 Results")
    mp3s_idxs = list(genre_activations.index)

    if playlist_option == "BPM":
        mp3s_idxs = st_helpers.filter_by_bpm(
            essentia_descriptors, max_bpm=max_tempo, min_bpm=min_tempo
        )
        st.success(f"Found {len(mp3s_idxs)} songs")
        st.dataframe(genre_activations.iloc[mp3s_idxs])

    if playlist_option == "Genre":
        # TODO: understand this code
        result = st_helpers.filter_by_style(
            genre_activations, style_select, style_select_range
        )
        st.write(result)
        mp3s_idxs = result.index
        st.success(f"Found {len(mp3s_idxs)} songs")

    if style_rank:
        ranked = st_helpers.rank_by_style(genre_activations, style_rank, mp3s_idxs)
        st.write("Applied ranking by audio style predictions.")
        st.write(ranked)
        mp3s_idxs = list(ranked.index)

    if max_tracks:
        mp3s_idxs = mp3s_idxs[:max_tracks]
        st.write("Using top", len(mp3s_idxs), "tracks from the results.")

    if shuffle:
        mp3s_idxs = st_helpers.shuffle_tracks(mp3s_idxs)
        st.write("Applied random shuffle.")

    if save_file and file_name:
        st_helpers.save_playlist(
            file_name,
            genre_activations.iloc[
                mp3s_idxs, genre_activations.columns.get_loc("file_path")
            ].tolist(),
            PLAYLIST_PATH,
        )
        st.success(f'Stored M3U playlist (local filepaths) to `{file_name+".m3u8"}`.')
    else:
        st.warning("Playlist not saved.")

    # st_helpers.display_results(audio_analysis, max_tracks)

    # getting a list of strings of the audio files that will be played
    mp3ss_audio_list = genre_activations.iloc[
        mp3s_idxs, genre_activations.columns.get_loc("file_path")
    ].tolist()
    st_helpers.display_audio_previews(mp3ss_audio_list)
