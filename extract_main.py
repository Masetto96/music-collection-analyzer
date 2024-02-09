import os
import logging

import essentia

import utils as u
from loader import AudioLoader
from extractor import FeatureExtractor

DATA_PATH = "audio"
DESCRIPTORS_PATH = "descriptors"
EMBEDDINGS_PATH = "embeddings"

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        filemode="w",
        filename="logs.log",
        datefmt="%H:%M:%S",
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    essentia.log.warningActive = False  # deactivate the warning level

    audio_loader = AudioLoader(data_path=DATA_PATH)
    feature_extractor = FeatureExtractor()

    all_features = []
    genre_activations = []
    for audio, sr, audio_mono, filename in audio_loader.yield_all():

        # get embeddings needed for music similarity and input to essentia models
        discogs_embeddings = feature_extractor.get_discogss_efnet_embeddings(audio_mono)
        music_cnn_embeddings = feature_extractor.get_msd_music_cnn_embeddings(
            audio_mono
        )
        # TODO: save embeddings

        genre = feature_extractor.predict_genre(discogs_embeddings)
        genre_activations.append(
            [filename.as_posix(), u.parse_discogs_genre_activations(genre)]
        )

        features = [
            filename.as_posix(),
            {
                "loudness": feature_extractor.extract_loudness(audio, sr),
                "tempo": feature_extractor.extract_tempo(audio_mono),
                "voice_instrumental": feature_extractor.predict_voice_instrumental(
                    discogs_embeddings
                ),
                "danceability": feature_extractor.predict_danceability(
                    discogs_embeddings
                ),
                "arousal_valence": feature_extractor.predict_arousal_valence(
                    music_cnn_embeddings
                ),
                "keyscale_edma": feature_extractor.extract_key_edma(audio_mono),
                "keyscale_krumhansl": feature_extractor.extract_key_krumhansl(
                    audio_mono
                ),
                "keyscale_temperly": feature_extractor.extract_key_temperly(audio_mono),
            },
        ]

        # Append features to the list
        all_features.append(features)

    # create a directory if not exists
    os.makedirs(DESCRIPTORS_PATH, exist_ok=True)
    u.save_result(DESCRIPTORS_PATH, "descriptors", all_features, pickle_only=True)
    u.save_result(DESCRIPTORS_PATH, "discogs-400-genre", all_features, pickle_only=True)
