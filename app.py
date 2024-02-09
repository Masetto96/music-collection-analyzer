"""
For tempo, implement search by tempo, where you can specify a tempo range (min to max BPM).

For voice/instrumental classifier, implement a binary selector that allows you to show only music with or without voice.

For danceability, implement a range selector. In the case of Essentia's signal processing algorithm, its output is within the [0, 3] range. In the case of the classifier, the output value is probability of music being danceable, within [0,1] range.

For arousal and valence, the range of values output by the model is within [1, 9]. Implement a range selector.

For key/scale, you can have a dropdown menu to select the key and select major vs minor scale. We have estimations done by three different profiles (`temperley`, `krumhansl`, or `edma`). Decide which of them you want to use for the UI.

"""

import streamlit as st
import utils as u

import streamlit_helpers as st_helpers

# ESSENTIA_ANALYSIS_PATH = 'data/files_essentia_effnet-discogs.jsonl.pickle'
ESSENTIA_ANALYSIS_PATH = 'descriptors/discogs-400-genre.pkl'

PLAYLIST_PATH = "playlists"

# Main script
st.write('# Audio analysis playlists example')
st.write(f'Using analysis data from `{ESSENTIA_ANALYSIS_PATH}`.')
audio_analysis = u.load_essentia_analysis(ESSENTIA_ANALYSIS_PATH)
audio_analysis_styles = audio_analysis.columns
st.write('Loaded audio analysis for', len(audio_analysis), 'tracks.')

st.write('## 🔍 Select')
st.write('### By style')
st.write('Style activation statistics:')
st.write(audio_analysis.describe())

style_select = st.multiselect('Select by style activations:', audio_analysis_styles)
if style_select:
    st.write(audio_analysis[style_select].describe())
    style_select_str = ', '.join(style_select)
    style_select_range = st.slider(f'Select tracks with `{style_select_str}` activations within range:', value=[0.5, 1.])

st.write('## 🔝 Rank')
style_rank = st.multiselect('Rank by style activations (multiplies activations for selected styles):', audio_analysis_styles, [])

st.write('## 🔀 Post-process')
max_tracks = st.number_input('Maximum number of tracks (0 for all):', value=0)
shuffle = st.checkbox('Random shuffle')

save_file = st.checkbox("Save Playlist?")
if save_file:
    file_name = st.text_input("Enter the name for the playlist:")

if st.button("RUN"):
    st.write('## 🔊 Results')
    mp3s = list(audio_analysis.index)

    if style_select:
        result = st_helpers.filter_by_style(audio_analysis, style_select, style_select_range)
        st.write(result)
        mp3s = result.index

    if style_rank:
        ranked = st_helpers.rank_by_style(audio_analysis, style_rank, mp3s)
        st.write('Applied ranking by audio style predictions.')
        st.write(ranked)
        mp3s = list(ranked.index)

    if max_tracks:
        mp3s = mp3s[:max_tracks]
        st.write('Using top', len(mp3s), 'tracks from the results.')

    if shuffle:
        mp3s = st_helpers.shuffle_tracks(mp3s)
        st.write('Applied random shuffle.')

    if save_file and file_name:
        st_helpers.save_playlist(file_name, mp3s, PLAYLIST_PATH)
        st.write(f'Stored M3U playlist (local filepaths) to `{file_name+".m3u8"}`.')
    else:
        st.write("File not saved.")

    st_helpers.display_results(audio_analysis, max_tracks)
    st_helpers.display_audio_previews(mp3s)

