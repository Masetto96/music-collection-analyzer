import logging

import essentia.standard as es
from pathlib import Path


class AudioLoader(object):
    """
    Only supports extensions listed below.

    - data_path (str): Path to the directory containing audio files.

    """

    def __init__(self, data_path: str):
        logging.debug("Initializing AudioLoader")
        self.data_path = Path(data_path)
        self.allowed_extensions = {".mp3", ".wav", ".flac", ".aac"}
        logging.debug("Looking for files in: %s", self.data_path)
        self.total_num_files_found = len(
            list(
                file
                for file in self.data_path.rglob("*")
                if file.suffix.lower() in self.allowed_extensions and file.is_file()
            )
        )
        logging.info("Found %d audio files to analyze", self.total_num_files_found)

    def _load_audio(self, filename: str):
        """
        Load audio from the given file.
        Returns stereo, with num of channels and samp rate, and mono version.

        Parameters:
        - filename (str): Path to the audio file.

        Returns:
        - audio (ndarray): Loaded audio data.
        - sr (int): Sample rate of the audio.
        - mono_audio (ndarray): Loaded audio data in mono format.
        """
        try:
            logging.debug("Loading file: %s", filename)
            audio, sr, nc, _, _, _ = es.AudioLoader(filename=filename.as_posix())()

            mono_audio = es.MonoMixer()(audio, nc)

            resampled_mono_audio = es.Resample(
                inputSampleRate=float(sr), outputSampleRate=float(16000), quality=1
            ).compute(mono_audio)

            return audio, sr, resampled_mono_audio

        except Exception as e:
            logging.error(e)

    def yield_all(self):
        """
        Generator function to yield loaded audio for all valid files in the specified directory.

        Yields:
        - Tuple: (audio, sr, audio_mono) for each valid audio file.
        """
        for file in self.data_path.rglob("*"):
            if file.suffix.lower() in self.allowed_extensions and file.is_file():
                yield *self._load_audio(file), file
