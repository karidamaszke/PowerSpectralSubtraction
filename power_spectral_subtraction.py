import numpy as np
from math import floor
from numpy.fft import fft, ifft


def get_power_spectral_density(signal):
    """
    Estimate power spectral density of given signal.
    psd = (1 / N) * |fft(signal)|^2    where    N - number of samples
                                                fft(.) - Fourier transform
                                                |.| - modulus -> sqrt(real^2 + imag^2)
    :param signal: np.array
    :return: power spectral density
    """

    signal_ft = fft(signal)
    return np.array([(1 / len(signal_ft)) * np.sqrt(signal_ft[i].real ** 2 + signal_ft[i].imag ** 2) for i in
                     range(len(signal_ft) // 2)])


class PowerSpectralSubtraction:
    """
    Class for demonstrating Power Spectral Subtraction algorithm and it's usage at denoising signals
    """

    def __init__(self, fs, signal):
        self.sample_rate = fs
        self.signal = signal

        self.frame_length = int(0.02 * self.sample_rate)  # 20 ms per one frame
        self.n_frames = floor(len(self.signal) / self.frame_length) - floor(self.sample_rate / self.frame_length)

        self.noise_psd = self.get_noise_psd()

    def get_noise_psd(self):
        """
        As first second of signal contains only noise,
        it's possible to estimate power spectral density of noise

        :return: average power spectral density of noise in first second
        """
        noise_frames = floor(self.sample_rate / self.frame_length)
        noise_psd = np.zeros(self.frame_length // 2)

        for frame in range(noise_frames):
            noise_psd += get_power_spectral_density(
                self.signal[frame * self.frame_length:self.frame_length + frame * self.frame_length])

        return noise_psd / noise_frames

    def denoise_signal(self):
        """
        Provide power spectral subtraction for each frame (20 ms):
            0. Start from first second of signal (there is only noise before)
            1. Estimate power spectral density of noised signal (psd_y)
            2. Estimate power spectral density of original signal (psd_x = psd_y - psd_noise)
            3. Design denoising filter
            4. Evaluate Fourier transform of denoised signal
            5. Inverse Fourier transform to obtain estmation of original, noiseless signal

        :return: estimation of original signal
        """
        self.normalize_signal()  # cut samples outside last frame
        offset = self.sample_rate  # 0. start after first second
        original_signal = []

        for frame in range(self.n_frames):
            # estimate psd of noisy signal
            psd_y = get_power_spectral_density(self.signal[offset:offset + self.frame_length])  # 1

            # estimate psd of original signal
            psd_x = psd_y - self.noise_psd
            psd_x = np.array([psd_x[i] if psd_x[i] > 0 else 0 for i in range(len(psd_x))])  # 2

            # create denoising filter
            filter_coeff = self.design_filter(psd_x, psd_y)  # 3

            # estimate Fourier transform of current frame
            y_ft = fft(self.signal[offset:offset + self.frame_length])
            x_ft = filter_coeff * y_ft

            # inverse Fourier transform
            x = ifft(x_ft)  # 4
            for sample in x:
                original_signal.append(sample.astype(np.int16))

            # skip to next frame
            offset += self.frame_length

        return np.array(original_signal)

    def design_filter(self, psd_x: np.array, psd_y: np.array):
        """
        Design linear filter for transform noisy signal to noiseless signal

        Amplitude of filter -> sqrt(psd_x / psd_y)  where:
        :param psd_x: power spectral density of  noiseless signal (psd_y - psd_noise)
        :param psd_y: power spectral density of noisy signal
        :return: np.array -> linear, symmetric filter
        """
        a = np.sqrt(psd_x / psd_y)
        a_rev = a.copy()
        a_rev = np.flipud(a_rev)
        return np.concatenate([a, a_rev])

    def normalize_signal(self):
        """
        Cut single samples at the end of signal
        """
        samples = len(self.signal)
        frames = samples // self.frame_length
        self.signal = self.signal[:frames * self.frame_length]
