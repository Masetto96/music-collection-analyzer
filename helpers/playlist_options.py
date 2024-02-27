# In playlist_functions.py
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def select_by_bpm(essentia_descriptors):
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
    return min_tempo, max_tempo

def select_by_arousal_valence(essentia_descriptors):
    st.write("### By Arousal and Valence")
    joint_plot = sns.jointplot(
        x=essentia_descriptors["valence_arousal"].apply(lambda x: x[0]),
        y=essentia_descriptors["valence_arousal"].apply(lambda x: x[1]),
        kind="hex",
        color="skyblue",
    )
    joint_plot.set_axis_labels("Valence", "Arousal", fontsize=11)
    min_arousal, max_arousal = st.slider(
        "Select Arousal Range:",
        min_value=1.0,
        max_value=9.0,
        value=(1.0, 9.0),
        step=0.01,
    )
    min_valence, max_valence = st.slider(
        "Select Valence Range:",
        min_value=1.0,
        max_value=9.0,
        value=(1.0, 9.0),
        step=0.01,
    )
    st.pyplot(joint_plot)
    return min_arousal, max_arousal, min_valence, max_valence

def select_vocal_instrumental(essentia_descriptors):
    st.write("### Vocal vs Instrumental")
    fig, ax = plt.subplots()
    ax.hist(essentia_descriptors["voice_instrumental"], color="skyblue", edgecolor="black", bins=20)
    ax.set_title("Distribution of vocal vs instrumental")
    ax.set_xlabel("Is instrumental")
    ax.set_ylabel("Counts")
    ax.grid(True)
    st.pyplot(fig)
    is_instrumental = st.toggle("Instrumental Tracks only, that is the value is bigger than 0.5")
    return is_instrumental

def select_by_danceability(essentia_descriptors):
    st.write("### By Danceability")
    fig, ax = plt.subplots()
    ax.hist(essentia_descriptors["danceability"], color="skyblue", edgecolor="black", bins=20)
    ax.set_title("Distribution of Danceability")
    ax.set_xlabel("Is danceable")
    ax.set_ylabel("Counts")
    ax.grid(True)
    st.pyplot(fig)
    lower_bound_dance, upper_bound_dance = st.slider(
        "Select a danceability range", 0.0, 1.0, (0.25, 0.75), step=0.01
    )
    return lower_bound_dance, upper_bound_dance

def select_by_key_scale(essentia_descriptors):
    st.write("### By Key and Scale")
    keys = set([x for x in essentia_descriptors["keyscale_edma"].apply(lambda x: x[0])])
    fig, ax = plt.subplots()
    profile_counts = essentia_descriptors["keyscale_edma"].apply(lambda x: (x[0], x[1])).value_counts()

    profile_counts.plot(kind='bar', ax=ax)
    ax.set_title("Distribution of keys and scales: edma profile")
    ax.set_xlabel("Key-Scale")
    ax.set_ylabel("Counts")
    ax.grid(True)
    ax.set_xticklabels(profile_counts.index, rotation=45, ha='right')
    st.pyplot(fig)

    # Create a dropdown menu
    key = st.selectbox("Choose a key", keys)
    mode = st.selectbox(
        'Choose a "mode"',
        (
            "major",
            "minor",
        ),
    )
    return key, mode
