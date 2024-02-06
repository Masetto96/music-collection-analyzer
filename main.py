import logging

from audio_loader import AudioLoader
from extractor import FeatureExtractor

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        filemode="w",
        filename="logs.log",
        datefmt="%H:%M:%S",
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    audio_loader = AudioLoader(data_path="audio_chunks")
    feature_extractor = FeatureExtractor()
    for audio, sr, audio_mono in audio_loader.yield_all():
        # to computed loudness (integrated LUFS), the algo needs stereo input
        # loudness = feature_extractor.extract_loudness(audio)

        bmp = feature_extractor.extract_tempo(audio_mono)
