import sys

import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd

from power_spectral_subtraction import PowerSpectralSubtraction


# ------------------------------------------------------------------------
PLAY = False  # set for play signals after separation
# ------------------------------------------------------------------------


def add_noise(original_signal):
    """
    Add artificial noise to signal. Noise is generated as random variable with normal distribution
    :param original_signal: np.array
    :return: noisy signal
    """
    noise = np.random.normal(0, 1, len(original_signal))
    return original_signal + noise


def main():
    try:
        fs, original_signal = wav.read("data\\original_signal.wav")
        noisy_signal = add_noise(original_signal)

        pss = PowerSpectralSubtraction(fs, noisy_signal)
        denoised_signal = pss.denoise_signal()

        wav.write("data\\denoised_signal.wav", fs, denoised_signal)

        if PLAY:
            sd.play(original_signal, fs, blocking=True)
            sd.play(noisy_signal, fs, blocking=True)
            sd.play(denoised_signal, fs, blocking=True)

    except Exception as e:
        print("Exception during algorithm! " + str(e))
        sys.exit(-1)


if __name__ == '__main__':
    main()
