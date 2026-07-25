import numpy as np
import librosa
import scipy.signal as signal


class VoiceProcessor:

    """
    Voice preprocessing pipeline for PC-GITA dataset.

    Extracted features:
    - MFCC
    - Jitter
    - Shimmer
    - HNR
    - Spectrogram
    """

    def __init__(
        self,
        sampling_rate=44100,
        mfcc_dim=40
    ):

        self.sr = sampling_rate
        self.mfcc_dim = mfcc_dim


    def load_audio(self, path):

        audio, sr = librosa.load(
            path,
            sr=self.sr
        )

        return audio, sr



    def extract_mfcc(self, audio):

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sr,
            n_mfcc=self.mfcc_dim
        )

        return np.mean(
            mfcc,
            axis=1
        )



    def extract_jitter(self, audio):

        """
        Approximation of fundamental frequency variation.
        """

        f0, _, _ = librosa.pyin(
            audio,
            fmin=50,
            fmax=500
        )


        f0=f0[~np.isnan(f0)]


        if len(f0)<2:
            return 0


        jitter=np.mean(
            np.abs(
                np.diff(f0)
            )
        ) / np.mean(f0)


        return jitter



    def extract_shimmer(self,audio):

        amplitude=np.abs(
            signal.hilbert(audio)
        )


        shimmer=np.mean(
            np.abs(
                np.diff(amplitude)
            )
        )


        return shimmer



    def extract_hnr(self,audio):

        noise=audio-signal.medfilt(
            audio,
            kernel_size=5
        )


        signal_power=np.mean(audio**2)

        noise_power=np.mean(noise**2)


        hnr=10*np.log10(
            signal_power /
            (noise_power+1e-8)
        )


        return hnr



    def extract_spectrogram(self,audio):

        spec=np.abs(
            librosa.stft(audio)
        )


        spec_db=librosa.power_to_db(
            spec**2
        )


        return spec_db



    def process(self,path):

        audio,_=self.load_audio(path)


        features={

            "MFCC":
            self.extract_mfcc(audio),

            "jitter":
            self.extract_jitter(audio),

            "shimmer":
            self.extract_shimmer(audio),

            "HNR":
            self.extract_hnr(audio),

            "spectrogram":
            self.extract_spectrogram(audio)

        }


        return features
