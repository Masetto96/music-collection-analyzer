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

    for audio, sr, audio_mono, filename in audio_loader.yield_all():
        # to computed loudness (integrated LUFS), the algo needs stereo input
        loudness = feature_extractor.extract_loudness(audio, sr)
        bmp = feature_extractor.extract_tempo(audio_mono)
        # key = feature_extractor.extract_key(audio_mono)

        # get embeddings needed for music similarity and input to essentia models
        music_cnn_embeddings = feature_extractor.get_msd_music_cnn_embeddings(
            audio_mono
        )
        discogs_embeddings = feature_extractor.get_discogss_efnet_embeddings(audio_mono)

        # compute descriptors from discogs embeddings
        voice_instrumental = feature_extractor.compute_voice_instrumental(
            discogs_embeddings
        )
        genre = feature_extractor.predict_genre(discogs_embeddings)
        danceability = feature_extractor.compute_danceability(discogs_embeddings)

        # compute descriptors from musicnn embeddings
        arousal_valence = feature_extractor.compute_arousal_valence(
            music_cnn_embeddings
        )
