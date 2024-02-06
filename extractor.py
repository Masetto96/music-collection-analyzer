import logging

import numpy as np
import essentia.standard as es


class FeatureExtractor(object):
    def __init__(
        self,
    ):
        """
        Initializes various feature extractors using pre-trained models from ESSENTIA.
        https://essentia.upf.edu/models.html
        """
        self.tempo_extractor = es.TempoCNN(graphFilename="weights/deeptemp-k4-3.pb")

        # https://essentia.upf.edu/reference/std_LoudnessEBUR128.html
        self.loudness_extractor = es.LoudnessEBUR128()

        self.discogs_efnet_embed = es.TensorflowPredictEffnetDiscogs(
            graphFilename="weights/discogs_multi_embeddings-effnet-bs64-1.pb",
            output="PartitionedCall:1",
        )
        self.msd_music_cnn_embeddings = es.TensorflowPredictMusiCNN(
            graphFilename="weights/msd-musicnn-1.pb", output="model/dense/BiasAdd"
        )
        self.discogs_genre_clf = es.TensorflowPredict2D(
            graphFilename="weights/genre_discogs400-discogs-effnet-1.pb",
            input="serving_default_model_Placeholder",
            output="PartitionedCall:0",
        )
        self.voice_instrumental_clf = es.TensorflowPredict2D(
            graphFilename="weights/voice_instrumental-discogs-effnet-1.pb",
            output="model/Softmax",
        )
        self.danceability_clf = es.TensorflowPredict2D(
            graphFilename="weights/danceability-discogs-effnet-1.pb",
            output="model/Softmax",
        )
        self.arousal_valence_clf = es.TensorflowPredict2D(
            graphFilename="weights/emomusic-msd-musicnn-2.pb", output="model/Identity"
        )

    def extract_tempo(self, audio: np.array):
        global_tempo, _, _ = self.tempo_extractor(audio)
        # returning beats per minute (bpm)
        return global_tempo

    def extract_key(self, audio: np.array):
        raise NotImplementedError

    def extract_loudness(self, audio: np.array, sr=44100):
        try:
            # for this loudness extractor, sampling rate is by default 44100
            assert int(sr) == 44100
            _, _, integrated_loudness, _ = self.loudness_extractor(audio)
            return integrated_loudness
        except Exception as e:
            # what to do in this case? resample!
            logging.warning(e)

    def get_discogss_efnet_embeddings(self, audio: np.array):
        return self.discogs_efnet_embed(audio)

    def get_msd_music_cnn_embeddings(self, audio: np.array):
        return self.msd_music_cnn_embeddings(audio)

    # TODO: look into how to average predictions
    def predict_genre(self, discogs_embeddings):
        predictions = self.discogs_genre_clf(discogs_embeddings)
        return predictions

    def compute_voice_instrumental(self, discogs_embeddings):
        predictions = self.voice_instrumental_clf(discogs_embeddings)
        return predictions

    def compute_danceability(self, discogs_embeddings):
        predictions = self.danceability_clf(discogs_embeddings)
        return predictions

    def compute_arousal_valence(self, music_cnn_embeddings):
        predictions = self.arousal_valence_clf(music_cnn_embeddings)
        return predictions
