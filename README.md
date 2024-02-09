This project is an assignment for the course:
# Music and Audio Processing Lab @ UPF, Music Technology Group

This repository contains scripts and tools for analyzing a collection of audio files using Essentia. This project aims to extract various audio descriptors from audio files, providing insights into their acoustic properties and facilitating tasks such as music classification, mood detection, or content-based recommendation systems.

# Setup

## Data and Weights
### Audio files shall be placed in the *audio* folder.

### The following weights need to be downloaded from essentia and placed in the *weights* folder. https://essentia.upf.edu/models.html

- danceability-discogs-effnet-1.pb
- discogs-effnet-bs64-1.pb
- emomusic-msd-musicnn-2.pb
- genre_discogs400-discogs-effnet-1.pb
- msd-musicnn-1.pb
- voice_instrumental-discogs-effnet-1.pb

# Usage
Create and activate a virtual environment:
```bash
python -m venv music-analyzer
source music-analyzer/bin/activate
```

Install requirements:
```bash
pip install requirements.txt
```
### To run the script to extract descriptors:
```bash
python extract_main.py
```

# Acknowledgments:
This project utilizes the Essentia library developed by the Music Technology Group at Universitat Pompeu Fabra in Barcelona, Spain. Special thanks to the Essentia team for their invaluable contribution to the field of audio analysis. https://essentia.upf.edu/