import os
import logging
import pickle
import essentia
import utils as u
from loader import AudioLoader
from extractor import FeatureExtractor
from tqdm import tqdm

DATA_PATH = "audio"
DESCRIPTORS_PATH = "descriptors"
EMBEDDINGS_PATH = "embeddings"
DISCOGS_EFFNET_METADATA_PATH = "metadata/discogs-effnet-bs64-1.json"

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
    feature_extractor = FeatureExtractor(
        discogs_effnet_metadata=DISCOGS_EFFNET_METADATA_PATH
    )

    # Create directories if  they not exist. 
    os.makedirs(EMBEDDINGS_PATH, exist_ok=True)
    os.makedirs(DESCRIPTORS_PATH, exist_ok=True)
    # Open file Streams
    embed_discogs_file = open(os.path.join(EMBEDDINGS_PATH, "discogs.pkl"), 'wb')
    embed_musiccnn_file = open(os.path.join(EMBEDDINGS_PATH, "musiccnn.pkl"), 'wb')
    embed_indexes = open(os.path.join(EMBEDDINGS_PATH, "embed_idxs.txt"), 'w')
    # discogs_json = open(os.path.join(DESCRIPTORS_PATH, "discogs-test.jsonl.pkl"), 'wb')

    counter  = 0
    all_features = []
    all_genre_activations = []
    for audio, sr, audio_mono, filename in tqdm(
        audio_loader.yield_all(), total=audio_loader.total_num_files_found
    ):
        # get embeddings needed for music similarity and input to essentia models
        discogs_embeddings = feature_extractor.get_discogss_efnet_embeddings(audio_mono)
        music_cnn_embeddings = feature_extractor.get_msd_music_cnn_embeddings(
            audio_mono
        )
        # genre_activations = feature_extractor.predict_genre(discogs_embeddings)
        
        # counter += 1
        # if counter == 3:
        #     break

        # features = [
        #     filename.as_posix(),
        #     {
        #         "loudness": feature_extractor.extract_loudness(audio, sr),
        #         "tempo": feature_extractor.extract_tempo(audio_mono),
        #         "voice_instrumental": feature_extractor.predict_voice_instrumental(
        #             discogs_embeddings
        #         ),
        #         "danceability": feature_extractor.predict_danceability(
        #             discogs_embeddings
        #         ),
        #         "arousal_valence": feature_extractor.predict_arousal_valence(
        #             music_cnn_embeddings
        #         ),
        #         "keyscale_edma": feature_extractor.extract_key_edma(audio_mono),
        #         "keyscale_krumhansl": feature_extractor.extract_key_krumhansl(
        #             audio_mono
        #         ),
        #         "keyscale_temperly": feature_extractor.extract_key_temperly(audio_mono),
        #     },
        # ]

        # SAVE FEATURES and GENRE ACTIVATIONS
        # all_features.append(features) # Append features to the list
        # discogs_json.write(pickle.dumps({"file_path" : filename.as_posix(), **genre_activations}) + b"\n") 

        # SAVE EMBEDDINGS and INDEXES
        pickle.dump(discogs_embeddings, embed_discogs_file) # Append discogs_embeddings
        pickle.dump(music_cnn_embeddings, embed_musiccnn_file) # Append musiccnn_embeddings
        embed_indexes.write(filename.as_posix())
        embed_indexes.write("\n")
        
        # all_genre_activations.append(genre_activations) # Append genre_activations)
        logging.debug("Features computed and appended")

    # close file streams
    embed_musiccnn_file.close()
    embed_discogs_file.close()
    # discogs_json.close()

    # u.save_result(
    #     DESCRIPTORS_PATH, "descriptors-but-genre-v2", all_features, pickle_only=True
    # )
    # u.save_result(
    #     DESCRIPTORS_PATH, "discogs-400-genre", all_genre_activations, pickle_only=True
    # )
