import logging
import json
import numpy as np
import essentia.standard as es
import utils as u


class FeatureExtractor(object):
    def __init__(self, discogs_effnet_metadata: str):
        """
        Initializes various feature extractors using pre-trained models from ESSENTIA as well as DSP techniques.
        https://essentia.upf.edu/models.html
        """
        self.discogs_effnet_metadata = u.load_json(discogs_effnet_metadata)

        # self.tempo_extractor = es.RhythmExtractor2013()
        self.bpm_extractor = es.TempoCNN(graphFilename="weights/deeptemp-k16-3.pb")
        self.loudness_extractor = es.LoudnessEBUR128()

        # https://essentia.upf.edu/reference/std_KeyExtractor.html
        self.key_extractor_temperley = es.KeyExtractor(profileType="temperley")
        self.key_extractor_krumhansl = es.KeyExtractor(profileType="krumhansl")
        self.key_extractor_edma = es.KeyExtractor(profileType="edma")

        self.discogs_effnet_embed = es.TensorflowPredictEffnetDiscogs(
            graphFilename="weights/discogs-effnet-bs64-1.pb",
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
        """
        https://essentia.upf.edu/reference/std_RhythmExtractor2013.html
        """
        # bpm = self.tempo_extractor(audio)[0]
        # return int(bpm)
        global_tempo, _, _ = self.bpm_extractor(audio)
        return global_tempo
        # global_tempo, _, _ = self.tempo_extractor(audio)
        # returning beats per minute (bpm)
        # return global_tempo

    def extract_key_temperly(self, audio: np.array):
        return self.key_extractor_temperley(audio)

    def extract_key_edma(self, audio: np.array):
        return self.key_extractor_edma(audio)

    def extract_key_krumhansl(self, audio: np.array):
        return self.key_extractor_krumhansl(audio)

    def extract_loudness(self, audio: np.array, sr=44100):
        """
        https://essentia.upf.edu/reference/std_LoudnessEBUR128.html
        """
        try:
            # for this loudness extractor, sampling rate is by default 44100
            assert int(sr) == 44100
            _, _, integrated_loudness, _ = self.loudness_extractor(audio)
            return integrated_loudness
        except Exception as e:
            # what to do in this case? resample!
            logging.warning(e)

    def get_discogss_efnet_embeddings(self, audio: np.array):
        return self.discogs_effnet_embed(audio)

    def get_msd_music_cnn_embeddings(self, audio: np.array):
        return self.msd_music_cnn_embeddings(audio)

    def predict_genre(self, discogs_embeddings):
        """
        Returns activations for all the 400 genre.
        It averages over all the frames.
        """
        return self._parse_discogs_genre_activations(
            np.mean(self.discogs_genre_clf(discogs_embeddings), axis=0)
        )

    def predict_voice_instrumental(self, discogs_embeddings):
        """
        Returns softmax.
        Avaraging over all the frames.
        """
        return tuple(np.mean(self.voice_instrumental_clf(discogs_embeddings), axis=0).tolist())

    def predict_danceability(self, discogs_embeddings):
        """
        Returns softmax.
        Avaraging over all the frames.
        """
        return tuple(np.mean(self.danceability_clf(discogs_embeddings), axis=0).tolist())

    def predict_arousal_valence(self, music_cnn_embeddings):
        """
        Returns softmax.
        Avaraging over all the frames.
        """
        return tuple(np.mean(self.arousal_valence_clf(music_cnn_embeddings), axis=0).tolist())

    def _parse_discogs_genre_activations(self, activations: np.array):
        """
        Takes as input the activations for all the 400 genre.
        Uses metadata to associate each activations with the corresponding genre.
        Returns human readable information :)
        """
        classes = self.discogs_effnet_metadata["classes"]

        assert len(classes) == len(activations)

        return dict(zip(classes, activations.tolist()))
