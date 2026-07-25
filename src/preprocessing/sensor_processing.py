import numpy as np

from scipy import signal


class SensorProcessor:


    """
    Wearable sensor preprocessing.

    Pipeline:

    - 5 second windowing
    - Wavelet denoising
    - FFT transformation
    - Motion feature extraction
    """



    def __init__(
        self,
        sampling_rate=100,
        window_seconds=5
    ):

        self.fs=sampling_rate

        self.window_size=(
            sampling_rate*
            window_seconds
        )



    def create_windows(
        self,
        data
    ):

        windows=[]


        for i in range(
            0,
            len(data)-self.window_size,
            self.window_size
        ):

            windows.append(
                data[
                    i:i+self.window_size
                ]
            )


        return np.array(windows)



    def wavelet_denoising(
        self,
        signal_data
    ):

        """
        Simple smoothing approximation.
        """

        return signal.savgol_filter(
            signal_data,
            11,
            3,
            axis=0
        )



    def fft_features(
        self,
        window
    ):


        fft=np.abs(
            np.fft.rfft(
                window,
                axis=0
            )
        )


        return fft



    def extract_motion_features(
        self,
        window
    ):


        rms=np.sqrt(
            np.mean(
                window**2,
                axis=0
            )
        )


        jerk=np.mean(
            np.diff(window,axis=0)**2
        )


        entropy=(
            -np.sum(
                window*np.log(
                    np.abs(window)+1e-8
                )
            )
        )


        return {

            "RMS":
            rms,

            "jerk":
            jerk,

            "entropy":
            entropy

        }



    def process(
        self,
        sensor_data
    ):

        windows=self.create_windows(
            sensor_data
        )


        results=[]


        for w in windows:


            w=self.wavelet_denoising(w)


            fft=self.fft_features(w)


            features=self.extract_motion_features(
                w
            )


            results.append(
                {
                    "window":w,
                    "fft":fft,
                    "features":features
                }
            )


        return results
