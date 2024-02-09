import random
import streamlit as st
import os

# Filtering module
def filter_by_style(audio_analysis, style_select, style_select_range):
    result = audio_analysis
    for style in style_select:
        result = result.loc[result[style] >= style_select_range[0]]
    return result

# Ranking module
def rank_by_style(audio_analysis, style_rank, mp3s):
    audio_analysis_query = audio_analysis.loc[mp3s][style_rank]
    audio_analysis_query['RANK'] = audio_analysis_query[style_rank[0]]
    for style in style_rank[1:]:
        audio_analysis_query['RANK'] *= audio_analysis_query[style]
    ranked = audio_analysis_query.sort_values(['RANK'], ascending=[False])
    ranked = ranked[['RANK'] + style_rank]
    return ranked

# Shuffling module
def shuffle_tracks(track_list):
    random.shuffle(track_list)
    return track_list

# Saving module
def save_playlist(file_name, track_list, playlist_path):
    with open(os.path.join(playlist_path, file_name+".m3u8"), 'w') as f:
        mp3_paths = [os.path.join('..', mp3) for mp3 in track_list]
        f.write('\n'.join(mp3_paths))

# Display results module
def display_results(result, max_tracks):
    st.write(result.head(max_tracks))

def display_audio_previews(track_list):
    st.write('Audio previews for the first 10 results:')
    for mp3 in track_list[:10]:
        st.audio(mp3, format="audio/mp3", start_time=0)
