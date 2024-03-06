import random
import streamlit as st
import matplotlib.pyplot as plt
import os
import helpers.utils as u


def filter_by_bpm(audio_analysis, max_bpm: int, min_bpm: int):
    return audio_analysis[(audio_analysis["tempo"] >= min_bpm) & (audio_analysis["tempo"] <= max_bpm)].index


def filter_by_arousal_and_valence(
    audio_analysis,
    max_valence: float,
    min_valence: float,
    max_arousal: float,
    min_arousal: float,
):
    idxs = (audio_analysis["valence_arousal"].apply(lambda x: x[1]).between(min_arousal, max_arousal)) & (
        audio_analysis["valence_arousal"].apply(lambda x: x[0]).between(min_valence, max_valence)
    )
    return audio_analysis[idxs].index


def filter_by_danceability(audio_analysis, danceability_upper: float, danceability_lower: float):
    return audio_analysis[
        (audio_analysis["danceability"] >= danceability_lower)
        & (audio_analysis["danceability"] <= danceability_upper)
    ].index


def filter_vocal_instrumental(audio_analysis, is_instrumental: bool = False):
    if is_instrumental:
        return audio_analysis[audio_analysis["voice_instrumental"] > 0.5].index
    else:
        return audio_analysis[audio_analysis["voice_instrumental"] <= 0.5].index


# Function to check if all words in a list are present in the tuple
def _contains_all_words(tuple_of_strings, words):
    return all(word in tuple_of_strings for word in words)


def filter_by_key_scale(audio_analysis, key: str, scale: str):
    # idxs = audio_analysis["keyscale_edma"]
    contains_words_mask = audio_analysis["keyscale_edma"].apply(
        lambda x: _contains_all_words(x, [key, scale])
    )
    return audio_analysis[contains_words_mask].index


# Ranking module
def rank_by_style(audio_analysis, style_rank, mp3s):
    audio_analysis_query = audio_analysis.loc[mp3s][style_rank]
    audio_analysis_query["RANK"] = audio_analysis_query[style_rank[0]]
    for style in style_rank[1:]:
        audio_analysis_query["RANK"] *= audio_analysis_query[style]
    ranked = audio_analysis_query.sort_values(["RANK"], ascending=[False])
    ranked = ranked[["RANK"] + style_rank]
    return ranked


# Shuffling module
def shuffle_tracks(track_list: list):
    random.shuffle(track_list)
    return track_list


# Saving module
def save_playlist(file_name: str, track_list: list, playlist_path):
    with open(os.path.join(playlist_path, file_name + ".m3u8"), "w") as f:
        mp3_paths = [os.path.join("..", mp3) for mp3 in track_list]
        f.write("\n".join(mp3_paths))


def display_audio_previews(track_list: list):
    st.info(" 🔊 Audio preview available only for 10 tracks.")
    for mp3 in track_list[:10]:
        st.write("\n".join(mp3))
        st.audio(mp3, format="audio/mp3", start_time=0)


def plot_parent_genre_distribution(discogs_df):
    # discard last column as it contains filepath
    max_column_per_row = discogs_df.loc[:, discogs_df.columns[:-1]].idxmax(axis="columns")

    # Counting only parent genre divided by "--"
    parent_genre_counts = max_column_per_row.apply(lambda x: x.split("---")[0]).value_counts()

    # Plot the bar chart
    fig, ax = plt.subplots()
    ax.bar(
        parent_genre_counts.index,
        parent_genre_counts.values,
        color="skyblue",
        edgecolor="black",
    )

    # Set labels and title
    ax.set_xlabel("Parent Genre")
    ax.set_ylabel("Counts")
    ax.set_title("Distribution of Parent Genres")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")

    # Display the plot using Streamlit
    st.pyplot(fig)
