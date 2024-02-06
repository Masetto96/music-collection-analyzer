import essentia.standard as es


class FeatureExtractor(object):
    def __init__(
        self,
    ):
        self.bmp_extractor = es.RhythmExtractor2013()
        self.loudness_extractor = es.LoudnessEBUR128()
        self.discogs_efnet_embed = es.TensorflowPredictEffnetDiscogs(
            graphFilename="discogs_multi_embeddings-effnet-bs64-1.pb",
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

    def extract_tempo(self, audio):
        # TODO: check the CNN based model
        bpm, _, _, _, _ = self.bmp_extractor(y)
        return bpm

    def extract_key(self, audio):
        raise NotImplementedError

    def extract_loudness(self, audio, sr=44100):
        # TODO: check the startAtZero parameter
        _, _, integrated_loudness, _ = self.loudness_extractor(audio)
        return integrated_loudness

    def get_discogss_efnet_embeddings(self, audio):
        return self.discogs_efnet_embed(audio)

    def get_msd_music_cnn_embeddings(self, audio):
        return self.msd_music_cnn_embeddings(audio)

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
