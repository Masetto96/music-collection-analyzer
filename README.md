This project is an assignment for the course:
## Music and Audio Processing Lab @ UPF, Music Technology Group

# Description:
This repository contains scripts and tools for analyzing a collection of audio files using Essentia. This project aims to extract various audio descriptors from audio files, providing insights into their acoustic properties and facilitating tasks such as music classification, mood detection, or content-based recommendation systems.

# Usage
Create and activate a virtual environment:
```console
python -m venv music-analyzer
source music-analyzer/bin/activate
```

Install requirements:
```console
pip install requirements.txt
```

## Data and Weights
- Data shall be placed in the *audio_chunks* folder.
- Weights need to be downloaded from essentia and placed in the *weights* folder. https://essentia.upf.edu/models.html


### Run the script:
```console
python main.py
```


# Acknowledgments:
This project utilizes the Essentia library developed by the Music Technology Group at Universitat Pompeu Fabra in Barcelona, Spain. Special thanks to the Essentia team for their invaluable contribution to the field of audio analysis. https://essentia.upf.edu/