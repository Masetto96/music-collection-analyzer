This project is an assignment for the course:
# Music and Audio Processing Lab @ UPF, Music Technology Group
## Feature Extraction and Embeddings: 

Utilizing Essentia, extracts features and embeddings from audio files that are used to "navigate" through the collection of files with the following apps.

- **Playlist Builder**: Web app for generating playlists based on user-defined audio descriptor queries, allowing users to create playlists with to specific characteristics.
- **Music Similarity Comparison**: Web app to explore audio similarity by computing cosine similarity of embeddings of songs.

## Setup:
###  Data and Weights
#### Audio files shall be placed in the *audio* folder.

####  The following weights need to be downloaded from essentia and placed in the *weights* folder. https://essentia.upf.edu/models.html

- danceability-discogs-effnet-1.pb
- discogs-effnet-bs64-1.pb
- emomusic-msd-musicnn-2.pb
- genre_discogs400-discogs-effnet-1.pb
- msd-musicnn-1.pb
- voice_instrumental-discogs-effnet-1.pb

### Requirements
It encouraged the usage of a virtual environment :innocent:
```bash
pip install -r requirements.txt
```

## Usage
### To run the script to extract descriptors:
```bash
python extract_main.py
```

### To run the app to create playlists based on sonic descriptors:
```bash
streamlit run playlist_app.py
```

### To run the app to compare songs based on embedding similarity:
```bash
streamlit run embeddings_app.py
```
